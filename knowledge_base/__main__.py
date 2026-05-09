import argparse
import sys
import json
from .config import Config
from .tracker import Tracker
from .retrieval import search


def cmd_stats(config: Config, args):
    tracker = Tracker(config)
    stats = tracker.get_stats()
    if args.format == "json":
        print(json.dumps(stats, indent=2))
    else:
        print("=== Learning Statistics ===")
        print(f"Sessions:    {stats['session_count']}")
        print(f"Questions:   {stats['question_count']}")
        print(f"Topics:      {stats['topic_count']}")
        print(f"Sources:     {stats['source_count']}")
        if stats["topics"]:
            print("\n=== Topics ===")
            for t in stats["topics"]:
                bar = "█" * t["mastery"] + "░" * (5 - t["mastery"])
                print(f"  [{bar}] {t['name']} ({t['question_count']} questions, last: {t['last_reviewed'] or 'never'})")


def cmd_topics(config: Config, args):
    tracker = Tracker(config)
    topics = tracker.list_topics()
    if args.format == "json":
        print(json.dumps(topics, indent=2))
    else:
        for t in topics:
            bar = "█" * t["mastery"] + "░" * (5 - t["mastery"])
            print(f"  [{bar}] {t['name']} (level {t['mastery']}/5, {t['question_count']} questions)")


def cmd_export(config: Config, args):
    tracker = Tracker(config)
    if args.topic:
        md = tracker.export_topic_markdown(args.topic)
        out_path = args.output or f"./exports/{args.topic.replace(' ', '_')}.md"
        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Exported to {out_path}")
    else:
        print("Specify a topic with --topic")


def cmd_search_cmd(config: Config, args):
    results = search(args.query, config=config)
    if not results:
        print("No results found.")
        return
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:500])


def main():
    parser = argparse.ArgumentParser(prog="knowledge_base", description="AI Personal Knowledge Base")
    sub = parser.add_subparsers(dest="command")

    p_stats = sub.add_parser("stats", help="Show learning progress")
    p_stats.add_argument("--format", choices=["table", "json"], default="table")

    p_topics = sub.add_parser("topics", help="List all topics")
    p_topics.add_argument("--format", choices=["table", "json"], default="table")

    p_export = sub.add_parser("export", help="Export topic summary as Markdown")
    p_export.add_argument("--topic", required=True)
    p_export.add_argument("--output")

    p_search = sub.add_parser("search", help="Search local knowledge base")
    p_search.add_argument("query")

    args = parser.parse_args()
    config = Config.from_env()

    if args.command == "stats":
        cmd_stats(config, args)
    elif args.command == "topics":
        cmd_topics(config, args)
    elif args.command == "export":
        cmd_export(config, args)
    elif args.command == "search":
        cmd_search_cmd(config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
