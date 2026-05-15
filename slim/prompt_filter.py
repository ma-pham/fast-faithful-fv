import sys
import traceback

from loguru import logger

from recipe.function_vectors.prompt_eval_filter import prompt_filter_main

if __name__ == "__main__":
    logger.info(f"args: {sys.argv[1:]}")
    try:
        prompt_filter_main(sys.argv[1:])
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
