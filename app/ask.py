"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, model: str = MODEL) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed = time.perf_counter() - started_at

    print(f"[応答時間] {elapsed:.2f}秒")

    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン数] {usage.prompt_token_count}")
        print(f"[出力トークン数] {usage.candidates_token_count}")

    return response.text


def list_models() -> None:
    client = genai.Client(api_key=get_api_key())
    models = [
        model
        for model in client.models.list()
        if model.name and model.supported_actions and "generateContent" in model.supported_actions
    ]
    if not models:
        print("生成に対応するモデルが見つかりませんでした。")
        return

    for model in models:
        print(f"{model.name} ({model.display_name or '名称なし'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=MODEL, help="使用するモデル名（既定: %(default)s）")
    parser.add_argument("--list-models", action="store_true", help="利用可能な生成モデルを一覧表示する")
    args = parser.parse_args()

    try:
        if args.list_models:
            list_models()
            return

        print(f"[model] {args.model}")
        print(f"[質問] {args.question}\n")
        print(ask(args.question, model=args.model))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
