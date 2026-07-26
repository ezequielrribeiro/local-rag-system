import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.generation.llm_client import LLMClient, NO_CONTEXT_MESSAGE
from src.retrieval.router import route_query
from src.retrieval.vector_store import HybridVectorStore

HISTORY_PATH = os.path.expanduser("~/.rag_history")


class REPL:
    def __init__(self, config: dict):
        self.config = config
        self.doc_type_mode = "auto"
        self.console = Console()

        emb_cfg = config["embedding"]
        llm_cfg = config["llm"]
        ret_cfg = config["retrieval"]
        vector_db_dir = config["paths"]["vector_db_dir"]

        self.store = HybridVectorStore(
            embedding_model_name=emb_cfg["model"],
            device=emb_cfg.get("device", "cpu"),
        )
        loaded = self.store.load(vector_db_dir)
        if not loaded:
            self.console.print(
                "[red]No vector store found.[/] Run [bold]ingest[/] first."
            )
            sys.exit(1)

        self.top_k = ret_cfg["top_k"]
        self.llm = LLMClient(
            endpoint=llm_cfg["endpoint"],
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )

        self.session = PromptSession(
            history=FileHistory(HISTORY_PATH),
        )

    def run(self):
        self._show_banner()
        while True:
            try:
                user_input = self.session.prompt("\n You> ")
            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                self.console.print("\n[yellow]Exiting.[/]")
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input.strip())
            else:
                self._handle_query(user_input.strip())

    def _show_banner(self):
        self.console.print(
            Panel.fit(
                "[bold cyan]RAG System — Interactive Mode[/]\n"
                f"Model: [green]{self.llm.model}[/] | "
                f"doc_type: [green]{self.doc_type_mode}[/]\n"
                "Type [bold]/help[/] for commands",
                border_style="cyan",
            )
        )

    def _handle_command(self, cmd: str):
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._show_help()
        elif command == "/clear":
            self.console.clear()
            self._show_banner()
        elif command == "/model":
            self._cmd_model(arg)
        elif command == "/doc-type":
            self._cmd_doc_type(arg)
        elif command in ("/quit", "/exit"):
            self.console.print("[yellow]Exiting.[/]")
            sys.exit(0)
        else:
            self.console.print(f"[red]Unknown command:[/] {command}  Type [bold]/help[/]")

    def _show_help(self):
        table = Table(show_header=False, box=None)
        table.add_column("Command", style="bold cyan", width=14)
        table.add_column("Description")
        table.add_row("/help", "Show this help")
        table.add_row("/clear", "Clear screen")
        table.add_row("/model", "Show current model")
        table.add_row("/model <name>", "Switch model (e.g. /model llama3.2)")
        table.add_row("/doc-type", "Show current doc_type filter")
        table.add_row("/doc-type <mode>", "Set filter: auto | user | tech | support")
        table.add_row("/quit", "Exit interactive mode")
        table.add_row("/exit", "Exit interactive mode")
        self.console.print(table)

    def _cmd_model(self, arg: str):
        if arg:
            self.llm.model = arg
        self.console.print(f"[green]Model:[/] {self.llm.model}")

    def _cmd_doc_type(self, arg: str):
        if arg:
            if arg not in ("auto", "user", "tech", "support"):
                self.console.print(
                    f"[red]Invalid doc_type:[/] {arg}  (use: auto | user | tech | support)"
                )
                return
            self.doc_type_mode = arg
        self.console.print(f"[green]doc_type:[/] {self.doc_type_mode}")

    def _handle_query(self, query: str):
        if self.doc_type_mode == "auto":
            filter_metadata = route_query(query)
        else:
            filter_metadata = {"doc_type": [self.doc_type_mode]}

        doc_type = (
            self.doc_type_mode
            if self.doc_type_mode != "auto"
            else (
                filter_metadata.get("doc_type", ["tech"])[0]
                if filter_metadata.get("doc_type")
                else "tech"
            )
        )

        with self.console.status("[bold green]Searching...[/]"):
            results = self.store.hybrid_search(
                query=query,
                top_k=self.top_k,
                filter_metadata=filter_metadata,
            )

        self.console.print()

        response_text = ""
        with Live(Markdown(""), refresh_per_second=12, vertical_overflow="visible") as live:
            for token in self.llm.generate_stream(query, results, doc_type):
                response_text += token
                live.update(Markdown(response_text or "..."))

        response_text = response_text.strip()
        if not response_text or response_text == NO_CONTEXT_MESSAGE:
            self.console.print(
                Panel(
                    "[yellow]Informação não encontrada na documentação técnica fornecida.[/]",
                    border_style="yellow",
                    title="Response",
                )
            )
        else:
            self.console.print(
                Panel(Markdown(response_text), border_style="green", title="Response")
            )

        if results:
            table = Table(show_header=True, box=None)
            table.add_column("#", style="dim", width=3)
            table.add_column("File", style="cyan")
            table.add_column("Source", style="dim")
            for i, c in enumerate(results, 1):
                table.add_row(str(i), c.metadata.filename, c.metadata.source)
            self.console.print(
                Panel(table, border_style="blue", title=f"Sources ({len(results)})")
            )
