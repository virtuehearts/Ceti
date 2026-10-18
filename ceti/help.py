"""help — the /help command text for the REPL.

One line per command. Kept here (not inline in cli.py) so the startup
screen, the REPL, and the docs all print the same list.
"""
from __future__ import annotations

HELP_TEXT = """\
CETI commands — ask anything, or use a slash command:

  /help              this list
  /tui               full-screen terminal UI (sessions, streaming)
  /quit              leave

  ask / answer
  /stats             tier stats, AIML + memory counts
  /trace             toggle the answer-ladder trace
  /tiers             which tiers answer each intent
  /oracle <question> skip memory/local, answer from the cloud oracle
  /benchmark         self-exam: quiz the brain on its own knowledge
  /evals             fixed benchmarks: coding / tool / knowledge
  /identity          node fingerprint + node id
  /heartbeat         toggle the idle research heartbeat

  memory / learning
  /tasks             pending agent tasks
  /agent             agent status

  tools / agent
  /tools             tool registry (all sandboxed)
  /call <tool> k=v   run one tool (\\n in values becomes a newline)
  /task <what>       cloud oracle drives tools for a job
  /tts [--voice NAME] <text>
                     speak text via the PocketTTS bridge
                     (default voice mya-03; WAV saved to work root/tts/)
  /web               start the ChatGPT-like web UI (background thread)
  /web stop          stop the web UI
  /skills            skill bundles (SKILL.md); /skills reload
  /ingest            drop files in data/library/, then ingest them
  /story ...         /story new|write|status|compile|list|article
  /bus ...           /bus post|read|compile — the memory bus (BIOS)

  network (signed, Gopher-style)
  /gopher            the Gopher menu
  /peers             known nodes
  /peer card|add     your identity card / file a peer
  /board             signed board; /board read|post|import
  /share <file>      sign + bundle a file
  /fetch <bundle>    verify + merge a signed bundle
  /nodelist          build|fetch|show the signed node list
  /gossip            export|import peer lists
  /chat <room>       live signed chat room
  /mail              inbox; /mail <fp> <subject> | <body>; /mail import
  /upload <file>     share + drop a bundle for others
  /download [file]   list the drop, or verify + merge one

  research (idle learning — see /research)
  /research               status: sites, topics, interests, budget
  /research sites [add|remove <domain>]
  /research topics [add|remove <topic>]
  /research interests [add|remove <text>]
  /research budget [max_topics N] [max_qa N] [max_minutes N]
"""
