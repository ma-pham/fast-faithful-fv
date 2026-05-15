import sys
import traceback

from loguru import logger

from recipe.function_vectors.cache_short_texts import main as cache_short_texts_main

if __name__ == "__main__":
    logger.info(f"args: {sys.argv[1:]}")
    try:
        cache_short_texts_main(sys.argv[1:])
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
