import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { complete, type Message } from "@mariozechner/pi-ai";
import {
	BorderedLoader,
	convertToLlm,
	serializeConversation,
	type ExtensionAPI,
	type ExtensionCommandContext,
	type SessionEntry,
	type Theme,
} from "@mariozechner/pi-coding-agent";
import { Key, matchesKey, truncateToWidth, visibleWidth } from "@mariozechner/pi-tui";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const PI_AGENT_DIR = process.env.PI_CODING_AGENT_DIR
	? path.resolve(expandHome(process.env.PI_CODING_AGENT_DIR))
	: path.join(os.homedir(), ".pi", "agent");

const DEFAULT_CONFIG = {
	maxMemoryCount: 20,
	maxInlineBytes: 50_000,
	showLoadedOnStartup: true,
};

type MemoryScope = "global" | "project";

type MemoryConfig = typeof DEFAULT_CONFIG;

const MEMORY_TAG_REGEX = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

type LoadedMemory = {
	scope: MemoryScope;
	name: string;
	relativePath: string;
	absolutePath: string;
	content: string;
	bytes: number;
	autoload: boolean;
	tags: string[];
};

type SkippedMemory = {
	scope: MemoryScope;
	name: string;
	relativePath: string;
	absolutePath: string;
	reason: string;
};

type MemoryState = {
	config: MemoryConfig;
	globalDir: string;
	projectRoot?: string;
	projectDir?: string;
	allMemories: LoadedMemory[];
	memories: LoadedMemory[];
	activeTags: string[];
	skipped: SkippedMemory[];
	errors: string[];
};

const secretPatterns: Array<{ label: string; regex: RegExp }> = [
	{ label: "private key", regex: /-----BEGIN [A-Z ]*PRIVATE KEY-----/ },
	{ label: "AWS access key", regex: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/ },
	{ label: "GitHub token", regex: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{30,}\b/ },
	{ label: "GitHub fine-grained token", regex: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/ },
	{ label: "OpenAI/Anthropic-style secret key", regex: /\bsk-[A-Za-z0-9_-]{20,}\b/ },
	{ label: "Slack token", regex: /\bxox[baprs]-[A-Za-z0-9-]{20,}\b/ },
];

const assignedSecretRegex = /\b(?:api[_-]?key|secret|token|password|credential)\b\s*[:=]\s*["']?([A-Za-z0-9_./+=-]{16,})/gi;

function expandHome(input: string): string {
	if (input === "~") return os.homedir();
	if (input.startsWith(`~${path.sep}`)) return path.join(os.homedir(), input.slice(2));
	return input;
}

async function pathExists(target: string): Promise<boolean> {
	try {
		await fs.access(target);
		return true;
	} catch {
		return false;
	}
}

async function isDirectory(target: string): Promise<boolean> {
	try {
		return (await fs.stat(target)).isDirectory();
	} catch {
		return false;
	}
}

async function readJsonConfig(file: string): Promise<Partial<MemoryConfig>> {
	try {
		const raw = await fs.readFile(file, "utf8");
		const parsed = JSON.parse(raw) as Partial<MemoryConfig>;
		return typeof parsed === "object" && parsed ? parsed : {};
	} catch {
		return {};
	}
}

function toPositiveInt(value: unknown, fallback: number): number {
	const parsed = typeof value === "string" ? Number(value) : typeof value === "number" ? value : Number.NaN;
	return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

async function findGitRoot(cwd: string): Promise<string | undefined> {
	let current = path.resolve(cwd);
	while (true) {
		if (await pathExists(path.join(current, ".git"))) return current;
		const parent = path.dirname(current);
		if (parent === current) return undefined;
		current = parent;
	}
}

async function findProjectMemoryLocation(cwd: string): Promise<{ root: string; dir: string } | undefined> {
	let current = path.resolve(cwd);
	const gitRoot = await findGitRoot(current);
	const stopAt = gitRoot ?? path.parse(current).root;

	while (true) {
		const candidate = path.join(current, ".pi", "memories");
		if (await isDirectory(candidate)) return { root: current, dir: candidate };
		if (current === stopAt) return undefined;
		const parent = path.dirname(current);
		if (parent === current) return undefined;
		current = parent;
	}
}

async function collectMarkdownFiles(dir: string, prefix = ""): Promise<string[]> {
	if (!(await isDirectory(dir))) return [];

	const entries = await fs.readdir(dir, { withFileTypes: true });
	const files: string[] = [];

	for (const entry of entries) {
		if (entry.name.startsWith(".")) continue;
		if (entry.name === "node_modules" || entry.name === ".git") continue;

		const relative = prefix ? path.join(prefix, entry.name) : entry.name;
		const absolute = path.join(dir, relative);

		if (entry.isDirectory()) {
			files.push(...(await collectMarkdownFiles(dir, relative)));
		} else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
			files.push(relative);
		}
	}

	return files.sort((a, b) => a.localeCompare(b));
}

function splitFrontmatter(raw: string): { frontmatter?: string; body: string } {
	if (!raw.startsWith("---\n") && !raw.startsWith("---\r\n")) return { body: raw.trim() };

	const normalized = raw.replace(/\r\n/g, "\n");
	const end = normalized.indexOf("\n---\n", 4);
	if (end === -1) return { body: raw.trim() };

	return {
		frontmatter: normalized.slice(4, end).trim(),
		body: normalized.slice(end + "\n---\n".length).trim(),
	};
}

function parseTitle(frontmatter: string | undefined): string | undefined {
	if (!frontmatter) return undefined;
	const line = frontmatter.split("\n").find((entry) => /^title\s*:/i.test(entry.trim()));
	if (!line) return undefined;
	return line
		.replace(/^title\s*:\s*/i, "")
		.trim()
		.replace(/^['\"]|['\"]$/g, "")
		.trim();
}

function unquoteYamlValue(value: string): string {
	return value.trim().replace(/^['\"]|['\"]$/g, "").trim();
}

function stripYamlComment(value: string): string {
	return value.replace(/\s+#.*$/, "").trim();
}

function parseInlineTags(value: string): string[] {
	const stripped = stripYamlComment(value);
	if (!stripped || stripped === "[]") return [];
	const inner = stripped.startsWith("[") && stripped.endsWith("]") ? stripped.slice(1, -1) : stripped;
	return inner
		.split(",")
		.map((tag) => unquoteYamlValue(tag))
		.filter(Boolean);
}

function parseMemoryMetadata(frontmatter: string | undefined): { autoload: boolean; tags: string[]; error?: string } {
	if (!frontmatter) return { autoload: true, tags: [] };

	const lines = frontmatter.split("\n");
	let autoload = true;
	let tags: string[] = [];

	for (let index = 0; index < lines.length; index++) {
		const trimmed = (lines[index] ?? "").trim();
		if (!trimmed || trimmed.startsWith("#")) continue;

		const autoloadMatch = trimmed.match(/^autoload\s*:\s*(.*)$/i);
		if (autoloadMatch) {
			const value = stripYamlComment(autoloadMatch[1]).toLowerCase();
			if (value === "true") autoload = true;
			else if (value === "false") autoload = false;
			else return { autoload: true, tags: [], error: `autoload must be true or false` };
			continue;
		}

		const tagsMatch = trimmed.match(/^tags\s*:\s*(.*)$/i);
		if (tagsMatch) {
			const value = tagsMatch[1].trim();
			if (value) {
				tags = parseInlineTags(value);
				continue;
			}

			const parsed: string[] = [];
			for (let next = index + 1; next < lines.length; next++) {
				const candidate = lines[next];
				if (!/^\s+/.test(candidate)) break;
				const listItem = candidate.trim().match(/^-\s*(.*)$/);
				if (!listItem) continue;
				const tag = unquoteYamlValue(stripYamlComment(listItem[1]));
				if (tag) parsed.push(tag);
				index = next;
			}
			tags = parsed;
		}
	}

	const seen = new Set<string>();
	const uniqueTags = tags.filter((tag) => {
		if (seen.has(tag)) return false;
		seen.add(tag);
		return true;
	});
	const invalid = uniqueTags.find((tag) => !MEMORY_TAG_REGEX.test(tag));
	if (invalid) return { autoload, tags: uniqueTags, error: `invalid tag "${invalid}"; tags must match ${MEMORY_TAG_REGEX.source}` };

	return { autoload, tags: uniqueTags };
}

function memoryName(relativePath: string, frontmatter?: string): string {
	const title = parseTitle(frontmatter);
	if (title) return title;
	return relativePath.replace(/\.md$/i, "");
}

function isEnvironmentVariableName(value: string): boolean {
	return /^[A-Z][A-Z0-9_]*$/.test(value) && value.includes("_");
}

function detectSecret(text: string): string | undefined {
	for (const pattern of secretPatterns) {
		pattern.regex.lastIndex = 0;
		if (pattern.regex.test(text)) return pattern.label;
	}

	assignedSecretRegex.lastIndex = 0;
	for (const match of text.matchAll(assignedSecretRegex)) {
		const value = match[1];
		if (value && !isEnvironmentVariableName(value)) return "assigned secret-like value";
	}

	return undefined;
}

async function loadMemoriesFromDir(scope: MemoryScope, dir: string): Promise<{ memories: LoadedMemory[]; skipped: SkippedMemory[]; errors: string[] }> {
	const memories: LoadedMemory[] = [];
	const skipped: SkippedMemory[] = [];
	const errors: string[] = [];

	for (const relativePath of await collectMarkdownFiles(dir)) {
		const absolutePath = path.join(dir, relativePath);
		try {
			const raw = await fs.readFile(absolutePath, "utf8");
			const { frontmatter, body } = splitFrontmatter(raw);
			const name = memoryName(relativePath, frontmatter);
			const secret = detectSecret(raw);
			const metadata = parseMemoryMetadata(frontmatter);

			if (secret) {
				skipped.push({
					scope,
					name,
					relativePath,
					absolutePath,
					reason: `suspected ${secret}`,
				});
				continue;
			}

			if (metadata.error) {
				skipped.push({
					scope,
					name,
					relativePath,
					absolutePath,
					reason: metadata.error,
				});
				continue;
			}

			memories.push({
				scope,
				name,
				relativePath,
				absolutePath,
				content: body,
				bytes: Buffer.byteLength(body, "utf8"),
				autoload: metadata.autoload,
				tags: metadata.tags,
			});
		} catch (error) {
			errors.push(`${absolutePath}: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	return { memories, skipped, errors };
}

async function loadConfig(projectRoot?: string): Promise<MemoryConfig> {
	const globalConfig = await readJsonConfig(path.join(PI_AGENT_DIR, "memory.config.json"));
	const projectConfig = projectRoot ? await readJsonConfig(path.join(projectRoot, ".pi", "memory.config.json")) : {};

	const merged = { ...DEFAULT_CONFIG, ...globalConfig, ...projectConfig };

	return {
		maxMemoryCount: toPositiveInt(process.env.PI_MEMORY_MAX_COUNT ?? merged.maxMemoryCount, DEFAULT_CONFIG.maxMemoryCount),
		maxInlineBytes: toPositiveInt(process.env.PI_MEMORY_MAX_INLINE_BYTES ?? merged.maxInlineBytes, DEFAULT_CONFIG.maxInlineBytes),
		showLoadedOnStartup: merged.showLoadedOnStartup !== false,
	};
}

async function loadState(cwd: string, activeTags: Iterable<string> = []): Promise<MemoryState> {
	const globalDir = path.join(PI_AGENT_DIR, "memories");
	await fs.mkdir(globalDir, { recursive: true });

	const projectLocation = await findProjectMemoryLocation(cwd);
	const config = await loadConfig(projectLocation?.root);

	const global = await loadMemoriesFromDir("global", globalDir);
	const project = projectLocation ? await loadMemoriesFromDir("project", projectLocation.dir) : { memories: [], skipped: [], errors: [] };
	const normalizedActiveTags = [...new Set([...activeTags].filter((tag) => MEMORY_TAG_REGEX.test(tag)))].sort((a, b) => a.localeCompare(b));
	const activeTagSet = new Set(normalizedActiveTags);
	const allMemories = [...global.memories, ...project.memories];

	return {
		config,
		globalDir,
		projectRoot: projectLocation?.root,
		projectDir: projectLocation?.dir,
		allMemories,
		memories: allMemories.filter((memory) => memory.autoload || memory.tags.some((tag) => activeTagSet.has(tag))),
		activeTags: normalizedActiveTags,
		skipped: [...global.skipped, ...project.skipped],
		errors: [...global.errors, ...project.errors],
	};
}

function groupByScope(memories: LoadedMemory[]): Record<MemoryScope, LoadedMemory[]> {
	return {
		global: memories.filter((memory) => memory.scope === "global"),
		project: memories.filter((memory) => memory.scope === "project"),
	};
}

function formatNames(memories: LoadedMemory[]): string {
	return memories.length === 0 ? "none" : memories.map((memory) => memory.name).join(", ");
}

function allTags(state: MemoryState): string[] {
	return [...new Set(state.allMemories.flatMap((memory) => memory.tags))].sort((a, b) => a.localeCompare(b));
}

function memoryKey(memory: LoadedMemory): string {
	return memory.absolutePath;
}

function isActiveMemory(state: MemoryState, memory: LoadedMemory): boolean {
	const active = new Set(state.memories.map(memoryKey));
	return active.has(memoryKey(memory));
}

function formatLoadedSummary(state: MemoryState): string {
	const grouped = groupByScope(state.memories);
	const manualAvailable = state.allMemories.filter((memory) => !memory.autoload && memory.tags.length > 0 && !isActiveMemory(state, memory)).length;
	const lines = [
		`Memories loaded (${state.memories.length})`,
		`Global: ${formatNames(grouped.global)}`,
		`Project: ${formatNames(grouped.project)}`,
	];

	if (state.activeTags.length > 0) lines.push(`Active manual tags: ${state.activeTags.join(", ")}`);
	if (manualAvailable > 0) lines.push(`Manual memories available: ${manualAvailable}. Use /memory tags and /memory load <tag>.`);

	if (state.skipped.length > 0) {
		lines.push(`Skipped: ${state.skipped.map((memory) => `${memory.name} (${memory.reason})`).join(", ")}`);
	}

	return lines.join("\n");
}

function relativeOrHome(absolutePath: string, cwd: string): string {
	const home = os.homedir();
	if (absolutePath.startsWith(`${home}${path.sep}`)) return `~${path.sep}${path.relative(home, absolutePath)}`;
	const relative = path.relative(cwd, absolutePath);
	if (relative && !relative.startsWith("..") && !path.isAbsolute(relative)) return relative;
	return absolutePath;
}

function formatMemoryTags(memory: LoadedMemory): string {
	return memory.tags.length > 0 ? memory.tags.join(", ") : "none";
}

function memoryStatus(state: MemoryState, memory: LoadedMemory): string {
	if (isActiveMemory(state, memory)) return memory.autoload ? "autoload" : "manual";
	return memory.tags.length > 0 ? "available" : "inactive";
}

class MemoryListModal {
	private scroll = 0;
	private maxScroll = 0;

	constructor(
		private state: MemoryState,
		private theme: Theme,
		private showAll: boolean,
		private done: () => void,
	) {}

	handleInput(data: string): void {
		if (matchesKey(data, Key.escape) || matchesKey(data, Key.enter) || matchesKey(data, "q")) {
			this.done();
			return;
		}

		if (matchesKey(data, Key.up)) this.scroll = Math.max(0, this.scroll - 1);
		else if (matchesKey(data, Key.down)) this.scroll = Math.min(this.maxScroll, this.scroll + 1);
		else if (matchesKey(data, Key.pageUp)) this.scroll = Math.max(0, this.scroll - 10);
		else if (matchesKey(data, Key.pageDown)) this.scroll = Math.min(this.maxScroll, this.scroll + 10);
		else if (matchesKey(data, Key.home)) this.scroll = 0;
		else if (matchesKey(data, Key.end)) this.scroll = this.maxScroll;
	}

	render(width: number): string[] {
		if (width <= 2) return [truncateToWidth(" ", width)];
		const outerWidth = Math.max(2, Math.min(width, 100));
		const innerWidth = outerWidth - 2;
		const rows = this.buildRows();
		const viewportRows = Math.min(rows.length, 18);
		this.maxScroll = Math.max(0, rows.length - viewportRows);
		this.scroll = Math.min(Math.max(0, this.scroll), this.maxScroll);

		const visibleRows = rows.slice(this.scroll, this.scroll + viewportRows);
		const title = this.showAll ? "Memory Files" : "Active Memories";
		const count = this.showAll ? `${this.state.memories.length}/${this.state.allMemories.length} active` : `${this.state.memories.length} active`;
		const lines = [
			this.border("top", innerWidth),
			this.row(` ${this.theme.fg("accent", this.theme.bold(title))} ${this.theme.fg("dim", count)}`, innerWidth),
			this.row("", innerWidth),
			...visibleRows.map((line) => this.row(line, innerWidth)),
		];

		if (this.maxScroll > 0) {
			lines.push(this.row(` ${this.theme.fg("dim", `showing ${this.scroll + 1}-${this.scroll + visibleRows.length} of ${rows.length}`)}`, innerWidth));
		}

		lines.push(this.row(` ${this.theme.fg("dim", "↑↓ scroll • Enter/Esc/q close")}`, innerWidth));
		lines.push(this.border("bottom", innerWidth));
		return lines;
	}

	invalidate(): void {}

	private buildRows(): string[] {
		const memories = this.showAll ? this.state.allMemories : this.state.memories;
		const grouped = groupByScope(memories);
		const rows: string[] = [];
		const manualAvailable = this.state.allMemories.filter((memory) => !memory.autoload && memory.tags.length > 0 && !isActiveMemory(this.state, memory)).length;

		if (this.state.activeTags.length > 0) rows.push(` ${this.theme.fg("dim", `Active manual tags: ${this.state.activeTags.join(", ")}`)}`);
		if (!this.showAll && manualAvailable > 0) rows.push(` ${this.theme.fg("dim", `${manualAvailable} manual memories available via /memory tags`)}`);
		if (!this.showAll && this.state.skipped.length > 0) rows.push(` ${this.theme.fg("warning", `${this.state.skipped.length} memories skipped; use /memory list --all for details`)}`);

		this.addScopeRows(rows, "Global", grouped.global);
		this.addScopeRows(rows, "Project", grouped.project);

		if (this.showAll && this.state.skipped.length > 0) {
			rows.push("");
			rows.push(` ${this.theme.fg("warning", this.theme.bold(`Skipped (${this.state.skipped.length})`))}`);
			for (const memory of this.state.skipped) {
				rows.push(`   ${this.theme.fg("warning", "○")} ${this.theme.fg("text", memory.name)} ${this.theme.fg("dim", `[${memory.scope}] ${memory.reason}`)}`);
			}
		}

		return rows.length > 0 ? rows : [` ${this.theme.fg("dim", "No memory files found.")}`];
	}

	private addScopeRows(rows: string[], title: string, memories: LoadedMemory[]): void {
		rows.push("");
		rows.push(` ${this.theme.fg("accent", this.theme.bold(`${title} (${memories.length})`))}`);

		if (memories.length === 0) {
			rows.push(`   ${this.theme.fg("dim", "none")}`);
			return;
		}

		for (const memory of memories) {
			const active = isActiveMemory(this.state, memory);
			const marker = active ? this.theme.fg("success", "●") : this.theme.fg("dim", "○");
			const tags = formatMemoryTags(memory);
			const tagsText = memory.tags.length > 0 ? this.theme.fg("muted", tags) : this.theme.fg("dim", tags);
			rows.push(`   ${marker} ${this.theme.fg("text", this.theme.bold(memory.name))} ${this.theme.fg("dim", `[${memoryStatus(this.state, memory)}] tags: `)}${tagsText}`);
		}
	}

	private border(position: "top" | "bottom", innerWidth: number): string {
		const [left, right] = position === "top" ? ["╭", "╮"] : ["╰", "╯"];
		return this.theme.fg("border", `${left}${"─".repeat(innerWidth)}${right}`);
	}

	private row(content: string, innerWidth: number): string {
		return `${this.theme.fg("border", "│")}${this.pad(content, innerWidth)}${this.theme.fg("border", "│")}`;
	}

	private pad(content: string, width: number): string {
		const truncated = truncateToWidth(content, width);
		return `${truncated}${" ".repeat(Math.max(0, width - visibleWidth(truncated)))}`;
	}
}

async function showMemoryListModal(state: MemoryState, ctx: ExtensionCommandContext, showAll: boolean): Promise<void> {
	await ctx.ui.custom<void>(
		(tui, theme, _keybindings, done) => {
			const modal = new MemoryListModal(state, theme, showAll, () => done(undefined));
			return {
				render: (width: number) => modal.render(width),
				invalidate: () => modal.invalidate(),
				handleInput: (data: string) => {
					modal.handleInput(data);
					tui.requestRender();
				},
			};
		},
		{
			overlay: true,
			overlayOptions: {
				width: "80%",
				minWidth: 60,
				maxHeight: "80%",
				margin: 2,
			},
		},
	);
}

function formatAllMemories(state: MemoryState, cwd: string): string {
	if (state.allMemories.length === 0) return "No memory files found.";
	const active = new Set(state.memories.map(memoryKey));
	return state.allMemories
		.map((memory) => {
			const status = active.has(memoryKey(memory))
				? memory.autoload
					? "loaded/autoload"
					: "loaded/manual"
				: memory.tags.length > 0
					? "available/manual"
					: "inactive/no-tags";
			return `- ${memory.name} [${memory.scope}, ${status}, tags: ${formatMemoryTags(memory)}] (${relativeOrHome(memory.absolutePath, cwd)})`;
		})
		.join("\n");
}

function formatMemoryTagsSummary(state: MemoryState): string {
	const tags = allTags(state);
	if (tags.length === 0) return "No memory tags found.";
	const active = new Set(state.memories.map(memoryKey));
	return tags
		.map((tag) => {
			const tagged = state.allMemories.filter((memory) => memory.tags.includes(tag));
			const loaded = tagged.filter((memory) => active.has(memoryKey(memory))).length;
			return `- ${tag}: ${loaded}/${tagged.length} loaded`;
		})
		.join("\n");
}

function buildMemoryIndex(state: MemoryState, cwd: string): string {
	const grouped = groupByScope(state.memories);
	const section = (title: string, memories: LoadedMemory[]) => {
		if (memories.length === 0) return `### ${title}\n\nNone.`;
		return `### ${title}\n\n${memories
			.map((memory) => `- ${memory.name} (${relativeOrHome(memory.absolutePath, cwd)})`)
			.join("\n")}`;
	};

	return `${section("Global Memories", grouped.global)}\n\n${section("Project Memories", grouped.project)}`;
}

function buildMemoryPrompt(state: MemoryState, cwd: string): string {
	const totalBytes = state.memories.reduce((sum, memory) => sum + memory.bytes, 0);
	const safety = `Memory safety: Never create, edit, or preserve hardcoded secrets in memory files. If credentials are relevant, refer to environment variable names only, never literal values.`;
	const behavior = `Treat active memories as durable user-provided context. Follow them unless they conflict with the user's current explicit instructions. Do not create, edit, or delete memory files unless the user explicitly asks.`;

	if (state.memories.length === 0) {
		return `\n\n## User Memories\n\nNo active memory files were loaded. ${safety}`;
	}

	if (totalBytes > state.config.maxInlineBytes) {
		return `\n\n## User Memories\n\n${behavior}\n${safety}\n\nActive memories exceed the inline budget (${totalBytes} bytes > ${state.config.maxInlineBytes} bytes), so only an index is included. Use the read tool to inspect relevant memory files when needed.\n\n${buildMemoryIndex(state, cwd)}`;
	}

	const grouped = groupByScope(state.memories);
	const renderScope = (title: string, memories: LoadedMemory[]) => {
		if (memories.length === 0) return `### ${title}\n\nNone.`;
		return `### ${title}\n\n${memories
			.map(
				(memory) =>
					`#### ${memory.name}\nSource: ${relativeOrHome(memory.absolutePath, cwd)}\n\n${memory.content || "(empty memory)"}`,
			)
			.join("\n\n")}`;
	};

	return `\n\n## User Memories\n\n${behavior}\n${safety}\n\n${renderScope("Global Memories", grouped.global)}\n\n${renderScope("Project Memories", grouped.project)}`;
}

function resolveToolPath(rawPath: unknown, cwd: string): string | undefined {
	if (typeof rawPath !== "string" || rawPath.trim() === "") return undefined;
	const withoutAt = rawPath.startsWith("@") ? rawPath.slice(1) : rawPath;
	return path.resolve(cwd, expandHome(withoutAt));
}

function isInside(candidate: string, directory: string): boolean {
	const relative = path.relative(path.resolve(directory), path.resolve(candidate));
	return relative === "" || (!!relative && !relative.startsWith("..") && !path.isAbsolute(relative));
}

function isMemoryPath(candidate: string | undefined, state: MemoryState): boolean {
	if (!candidate) return false;
	const dirs = [state.globalDir, state.projectDir].filter(Boolean) as string[];
	return dirs.some((dir) => isInside(candidate, dir));
}

function extractWriteText(toolName: string, input: unknown): string | undefined {
	if (!input || typeof input !== "object") return undefined;
	const record = input as Record<string, unknown>;

	if (toolName === "write" && typeof record.content === "string") return record.content;

	if (toolName === "edit" && Array.isArray(record.edits)) {
		return record.edits
			.map((edit) => (edit && typeof edit === "object" && typeof (edit as Record<string, unknown>).newText === "string" ? String((edit as Record<string, unknown>).newText) : ""))
			.join("\n");
	}

	return undefined;
}

type MemoryCommand =
	| { kind: "list"; all: boolean }
	| { kind: "reload" | "tags" }
	| { kind: "load" | "unload"; tag: string }
	| { kind: "add"; scope: MemoryScope; name: string; instructions?: string }
	| { kind: "edit"; scope: MemoryScope; name: string }
	| { kind: "help"; error?: string };

function parseTagArg(raw: string, action: string): MemoryCommand {
	const tag = raw.trim();
	if (!tag) return { kind: "help", error: `Missing tag.` };
	if (!MEMORY_TAG_REGEX.test(tag)) return { kind: "help", error: `Invalid tag "${tag}". Tags must match ${MEMORY_TAG_REGEX.source}.` };
	return { kind: action as "load" | "unload", tag };
}

function parseMemoryCommand(args: string): MemoryCommand {
	const trimmed = args.trim();
	if (!trimmed || trimmed === "list") return { kind: "list", all: false };
	if (trimmed === "list --all") return { kind: "list", all: true };
	if (trimmed === "reload") return { kind: "reload" };
	if (trimmed === "tags") return { kind: "tags" };

	const tagMatch = trimmed.match(/^(load|unload)\s+([\s\S]+)$/i);
	if (tagMatch) return parseTagArg(tagMatch[2], tagMatch[1].toLowerCase());

	const match = trimmed.match(/^(add|edit)\s+(global|project)\s+([\s\S]+)$/i);
	if (!match) return { kind: "help", error: `Unknown memory command: ${trimmed}` };

	const kind = match[1].toLowerCase() as "add" | "edit";
	const scope = match[2].toLowerCase() as MemoryScope;
	const rest = match[3].trim();
	if (!rest) return { kind: "help", error: `Missing memory name.` };

	if (kind === "edit") {
		const name = rest.split(/\s+--\s+/, 1)[0].trim();
		return name ? { kind, scope, name } : { kind: "help", error: `Missing memory name.` };
	}

	const parts = rest.split(/\s+--\s+/);
	const name = (parts.shift() ?? "").trim();
	const instructions = parts.join(" -- ").trim();
	return name ? { kind, scope, name, instructions: instructions || undefined } : { kind: "help", error: `Missing memory name.` };
}

function slugifyMemoryName(name: string): string {
	const withoutExtension = name.replace(/\.md$/i, "");
	const slug = withoutExtension
		.normalize("NFKD")
		.replace(/[\\/]+/g, " ")
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/-{2,}/g, "-");
	return slug ? `${slug}.md` : "";
}

function titleFromName(name: string): string {
	return name
		.replace(/\.md$/i, "")
		.replace(/[\\/_-]+/g, " ")
		.trim()
		.split(/\s+/)
		.map((word) => (word.toUpperCase() === word ? word : word.charAt(0).toUpperCase() + word.slice(1)))
		.join(" ");
}

function yamlDoubleQuote(value: string): string {
	return JSON.stringify(value);
}

function buildBlankMemoryTemplate(name: string): string {
	return `---\ntitle: ${yamlDoubleQuote(titleFromName(name))}\nautoload: true\ntags: []\n---\n\n`;
}

function detectDisallowedFrontmatter(raw: string): string | undefined {
	const { frontmatter } = splitFrontmatter(raw);
	if (!frontmatter) return undefined;
	const badLine = frontmatter.split("\n").find((line) => /^(scope|priority)\s*:/i.test(line.trim()));
	return badLine ? badLine.trim().split(":", 1)[0] : undefined;
}

function validateMemoryText(raw: string): string | undefined {
	const secret = detectSecret(raw);
	if (secret) return `Refusing to save memory because it appears to contain a ${secret}. Use environment variable names instead of literal secret values.`;
	const disallowed = detectDisallowedFrontmatter(raw);
	if (disallowed) return `Refusing to save memory because frontmatter must not include ${disallowed}. Use autoload and tags for conditional loading; global/project scope is inferred from file location.`;
	const { frontmatter } = splitFrontmatter(raw);
	const metadata = parseMemoryMetadata(frontmatter);
	if (metadata.error) return `Refusing to save memory because frontmatter is invalid: ${metadata.error}.`;
	return undefined;
}

async function determineProjectMemoryLocationForCreate(cwd: string): Promise<{ root: string; dir: string }> {
	const existing = await findProjectMemoryLocation(cwd);
	if (existing) return existing;
	const root = (await findGitRoot(cwd)) ?? path.resolve(cwd);
	return { root, dir: path.join(root, ".pi", "memories") };
}

async function getMemoryDir(scope: MemoryScope, ctx: ExtensionCommandContext, currentState: MemoryState, forCreate: boolean): Promise<string | undefined> {
	if (scope === "global") return currentState.globalDir;

	if (currentState.projectDir) return currentState.projectDir;
	if (!forCreate) {
		ctx.ui.notify("No project memory directory is active for this cwd.", "error");
		return undefined;
	}

	const location = await determineProjectMemoryLocationForCreate(ctx.cwd);
	const ok = await ctx.ui.confirm(
		"Create project memories?",
		`No project memory directory exists. Create ${relativeOrHome(location.dir, ctx.cwd)}?`,
	);
	if (!ok) return undefined;
	await fs.mkdir(location.dir, { recursive: true });
	return location.dir;
}

function findMemoryByName(currentState: MemoryState, scope: MemoryScope, name: string): { absolutePath: string; name: string } | undefined {
	const filename = slugifyMemoryName(name);
	const normalized = name.trim().toLowerCase();
	const loaded = currentState.allMemories
		.filter((memory) => memory.scope === scope)
		.find(
			(memory) =>
				path.basename(memory.relativePath).toLowerCase() === filename ||
				memory.relativePath.replace(/\.md$/i, "").toLowerCase() === normalized ||
				memory.name.toLowerCase() === normalized,
		);
	if (loaded) return { absolutePath: loaded.absolutePath, name: loaded.name };

	const skipped = currentState.skipped
		.filter((memory) => memory.scope === scope)
		.find(
			(memory) =>
				path.basename(memory.relativePath).toLowerCase() === filename ||
				memory.relativePath.replace(/\.md$/i, "").toLowerCase() === normalized ||
				memory.name.toLowerCase() === normalized,
		);
	if (skipped) return { absolutePath: skipped.absolutePath, name: skipped.name };

	return undefined;
}

function entryToMessage(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "message") return entry.message;
	if (entry.type === "compaction") {
		return {
			role: "compactionSummary",
			summary: entry.summary,
			tokensBefore: entry.tokensBefore,
			timestamp: new Date(entry.timestamp).getTime(),
		};
	}
	return undefined;
}

function getConversationText(ctx: ExtensionCommandContext): string {
	const messages = ctx.sessionManager.getBranch().map(entryToMessage).filter((message) => message !== undefined);
	const text = serializeConversation(convertToLlm(messages));
	const maxChars = 40_000;
	return text.length > maxChars ? `[Earlier conversation omitted for length.]\n\n${text.slice(-maxChars)}` : text;
}

const MEMORY_DRAFT_SYSTEM_PROMPT = `You draft durable Markdown memory files for a coding agent.

Rules:
- Output only the complete Markdown file content. No preamble.
- Include only stable, reusable context the user wants remembered.
- Be concise and specific.
- Use optional frontmatter with title, autoload, and tags only if helpful.
- autoload defaults to true; set autoload: false only for memories that should be manually loaded by tag.
- Tags must be lowercase letters/numbers with single hyphens between words (for example: testing, api-v2).
- Do not include scope or priority in frontmatter.
- Never include API keys, tokens, passwords, private keys, or literal secrets.
- If credentials are relevant, mention environment variable names only.
- Avoid temporal phrasing like "today", "just now", "above", or "this conversation" unless truly necessary.
- Current explicit user instructions should guide what is worth remembering.`;

function stripOuterMarkdownFence(text: string): string {
	const trimmed = text.trim();
	const match = trimmed.match(/^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$/i);
	return match ? match[1].trim() : trimmed;
}

async function generateMemoryDraft(command: Extract<MemoryCommand, { kind: "add" }>, filename: string, ctx: ExtensionCommandContext): Promise<string | undefined> {
	if (!ctx.model) {
		ctx.ui.notify("No model selected; opening a blank memory template instead.", "warning");
		return undefined;
	}

	const conversationText = getConversationText(ctx);
	const userMessage: Message = {
		role: "user",
		content: [
			{
				type: "text",
				text: [
					`Memory scope: ${command.scope}`,
					`Memory name: ${command.name}`,
					`Filename: ${filename}`,
					`User drafting instructions: ${command.instructions}`,
					"",
					"Conversation context:",
					conversationText || "(No prior conversation context available.)",
				].join("\n"),
			},
		],
		timestamp: Date.now(),
	};

	const result = await ctx.ui.custom<string | undefined>((tui, theme, _kb, done) => {
		const loader = new BorderedLoader(tui, theme, `Drafting memory with ${ctx.model!.id}...`);
		loader.onAbort = () => done(undefined);

		const run = async () => {
			const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model!);
			if (!auth.ok || !auth.apiKey) throw new Error(auth.ok ? `No API key for ${ctx.model!.provider}` : auth.error);

			const response = await complete(
				ctx.model!,
				{ systemPrompt: MEMORY_DRAFT_SYSTEM_PROMPT, messages: [userMessage] },
				{ apiKey: auth.apiKey, headers: auth.headers, signal: loader.signal },
			);

			if (response.stopReason === "aborted") return undefined;
			return response.content
				.filter((content): content is { type: "text"; text: string } => content.type === "text")
				.map((content) => content.text)
				.join("\n")
				.trim();
		};

		run()
			.then(done)
			.catch((error) => {
				console.error("Memory draft generation failed:", error);
				done(undefined);
			});

		return loader;
	});

	if (!result) ctx.ui.notify("Memory draft generation cancelled or failed; opening a blank template instead.", "warning");
	return result ? stripOuterMarkdownFence(result) : undefined;
}

async function editUntilValid(title: string, initialText: string, ctx: ExtensionCommandContext): Promise<string | undefined> {
	let text = initialText;
	while (true) {
		const edited = await ctx.ui.editor(title, text);
		if (edited === undefined) return undefined;

		const error = validateMemoryText(edited);
		if (!error) return edited.endsWith("\n") ? edited : `${edited}\n`;

		ctx.ui.notify(error, "error");
		text = edited;
	}
}

async function handleAddMemory(command: Extract<MemoryCommand, { kind: "add" }>, ctx: ExtensionCommandContext, currentState: MemoryState): Promise<MemoryState> {
	const filename = slugifyMemoryName(command.name);
	if (!filename) {
		ctx.ui.notify("Memory name must contain at least one letter or number.", "error");
		return currentState;
	}

	const dir = await getMemoryDir(command.scope, ctx, currentState, true);
	if (!dir) return currentState;

	await fs.mkdir(dir, { recursive: true });
	const target = path.join(dir, filename);
	if (await pathExists(target)) {
		ctx.ui.notify(`Memory already exists: ${relativeOrHome(target, ctx.cwd)}. Use /memory edit ${command.scope} ${command.name}.`, "error");
		return currentState;
	}

	let initialText = buildBlankMemoryTemplate(command.name);
	if (command.instructions) {
		initialText = (await generateMemoryDraft(command, filename, ctx)) || initialText;
	}

	const edited = await editUntilValid(`Edit ${command.scope} memory: ${command.name}`, initialText, ctx);
	if (edited === undefined) {
		ctx.ui.notify("Cancelled", "info");
		return currentState;
	}

	await fs.writeFile(target, edited, "utf8");
	const nextState = await loadState(ctx.cwd, currentState.activeTags);
	ctx.ui.notify(`Created ${command.scope} memory: ${relativeOrHome(target, ctx.cwd)}`, "info");
	return nextState;
}

async function handleEditMemory(command: Extract<MemoryCommand, { kind: "edit" }>, ctx: ExtensionCommandContext, currentState: MemoryState): Promise<MemoryState> {
	const match = findMemoryByName(currentState, command.scope, command.name);
	if (!match) {
		ctx.ui.notify(`No ${command.scope} memory found named "${command.name}".`, "error");
		return currentState;
	}

	const original = await fs.readFile(match.absolutePath, "utf8");
	const edited = await editUntilValid(`Edit ${command.scope} memory: ${match.name}`, original, ctx);
	if (edited === undefined) {
		ctx.ui.notify("Cancelled", "info");
		return currentState;
	}
	if (edited === original || edited === (original.endsWith("\n") ? original : `${original}\n`)) {
		ctx.ui.notify("No memory changes saved.", "info");
		return currentState;
	}

	await fs.writeFile(match.absolutePath, edited, "utf8");
	const nextState = await loadState(ctx.cwd, currentState.activeTags);
	ctx.ui.notify(`Updated ${command.scope} memory: ${relativeOrHome(match.absolutePath, ctx.cwd)}`, "info");
	return nextState;
}

function memoryUsage(): string {
	return [
		"Usage:",
		"/memory list",
		"/memory list --all",
		"/memory tags",
		"/memory load <tag>",
		"/memory unload <tag>",
		"/memory reload",
		"/memory add global <name>",
		"/memory add project <name>",
		"/memory add global <name> -- <drafting instructions>",
		"/memory add project <name> -- <drafting instructions>",
		"/memory edit global <name>",
		"/memory edit project <name>",
	].join("\n");
}

async function loadMemoryTag(tag: string, ctx: ExtensionCommandContext, currentState: MemoryState): Promise<MemoryState> {
	const before = new Set(currentState.memories.map(memoryKey));
	const matching = currentState.allMemories.filter((memory) => memory.tags.includes(tag));
	if (matching.length === 0) {
		ctx.ui.notify(`No memories found for tag "${tag}".`, "info");
		return currentState;
	}

	const loadable = matching.filter((memory) => !before.has(memoryKey(memory)));
	if (loadable.length === 0) {
		ctx.ui.notify(`No unloaded memories found for tag "${tag}".`, "info");
		return currentState;
	}

	const nextTags = [...new Set([...currentState.activeTags, tag])];
	const nextState = await loadState(ctx.cwd, nextTags);
	const newlyLoaded = nextState.memories.filter((memory) => memory.tags.includes(tag) && !before.has(memoryKey(memory)));

	ctx.ui.notify(
		[`Loaded ${newlyLoaded.length} memories for tag "${tag}":`, ...newlyLoaded.map((memory) => `- ${memory.name} (${relativeOrHome(memory.absolutePath, ctx.cwd)})`)].join("\n"),
		"info",
	);
	return nextState;
}

async function unloadMemoryTag(tag: string, ctx: ExtensionCommandContext, currentState: MemoryState): Promise<MemoryState> {
	if (!currentState.activeTags.includes(tag)) {
		ctx.ui.notify(`Tag "${tag}" is not currently loaded.`, "info");
		return currentState;
	}

	const nextState = await loadState(
		ctx.cwd,
		currentState.activeTags.filter((activeTag) => activeTag !== tag),
	);
	ctx.ui.notify(`Unloaded manual tag "${tag}". Autoload memories remain active.`, "info");
	return nextState;
}

export default function memoryExtension(pi: ExtensionAPI) {
	let state: MemoryState | undefined;

	pi.on("session_start", async (_event, ctx) => {
		state = await loadState(ctx.cwd);

		if (!ctx.hasUI) return;

		if (state.config.showLoadedOnStartup) {
			ctx.ui.notify(formatLoadedSummary(state), "info");
		}

		if (state.memories.length > state.config.maxMemoryCount) {
			ctx.ui.notify(
				`Memory warning: ${state.memories.length} memories are active, above the configured maxMemoryCount of ${state.config.maxMemoryCount}. Consider consolidating or deleting old memory files.`,
				"warning",
			);
		}

		for (const skipped of state.skipped) {
			const safetySuffix = skipped.reason.startsWith("suspected ") ? " Store secrets in environment variables, not memory files." : "";
			ctx.ui.notify(`Memory skipped: ${skipped.name} (${skipped.reason}).${safetySuffix}`, "warning");
		}

		for (const error of state.errors) {
			ctx.ui.notify(`Memory load error: ${error}`, "warning");
		}
	});

	pi.on("before_agent_start", async (event, ctx) => {
		state ??= await loadState(ctx.cwd);
		return { systemPrompt: event.systemPrompt + buildMemoryPrompt(state, ctx.cwd) };
	});

	pi.on("tool_call", async (event, ctx) => {
		if (!state) return;

		if (event.toolName === "write" || event.toolName === "edit") {
			const input = event.input as Record<string, unknown>;
			const target = resolveToolPath(input?.path, ctx.cwd);
			const text = extractWriteText(event.toolName, input);
			const secret = text ? detectSecret(text) : undefined;

			if (isMemoryPath(target, state) && secret) {
				return {
					block: true,
					reason: `Blocked write to memory file because it appears to contain a ${secret}. Store sensitive values in environment variables and reference the variable name in memory instead.`,
				};
			}
		}

		if (event.toolName === "bash") {
			const command = (event.input as { command?: unknown })?.command;
			if (typeof command !== "string") return;
			const memoryPathFragments = [state.globalDir, state.projectDir]
				.filter(Boolean)
				.flatMap((dir) => [dir!, path.relative(ctx.cwd, dir!)])
				.filter((fragment) => fragment.length > 0);
			const touchesMemory = memoryPathFragments.some((fragment) => command.includes(fragment)) || command.includes(".pi/memories");
			const secret = touchesMemory ? detectSecret(command) : undefined;
			if (touchesMemory && secret) {
				return {
					block: true,
					reason: `Blocked bash command because it appears to write a ${secret} to memory. Store sensitive values in environment variables and reference the variable name in memory instead.`,
				};
			}
		}
	});

	pi.registerCommand("memory", {
		description: "List, load, reload, add, or edit global/project memories",
		handler: async (args, ctx) => {
			const command = parseMemoryCommand(args);

			if (command.kind === "help") {
				ctx.ui.notify(`${command.error ? `${command.error}\n\n` : ""}${memoryUsage()}`, "info");
				return;
			}

			state ??= await loadState(ctx.cwd);

			if (command.kind === "reload") {
				state = await loadState(ctx.cwd, state.activeTags);
				ctx.ui.notify(`Reloaded memories.\n${formatLoadedSummary(state)}`, "info");
				return;
			}

			if (command.kind === "list") {
				if (ctx.hasUI) await showMemoryListModal(state, ctx, command.all);
				else ctx.ui.notify(command.all ? formatAllMemories(state, ctx.cwd) : formatLoadedSummary(state), "info");
				return;
			}

			if (command.kind === "tags") {
				ctx.ui.notify(formatMemoryTagsSummary(state), "info");
				return;
			}

			if (command.kind === "load") {
				state = await loadMemoryTag(command.tag, ctx, state);
				return;
			}

			if (command.kind === "unload") {
				state = await unloadMemoryTag(command.tag, ctx, state);
				return;
			}

			if (!ctx.hasUI) {
				ctx.ui.notify("Adding and editing memories requires interactive UI.", "error");
				return;
			}

			if (command.kind === "add") {
				state = await handleAddMemory(command, ctx, state);
				return;
			}

			if (command.kind === "edit") {
				state = await handleEditMemory(command, ctx, state);
				return;
			}
		},
	});
}
