import sys
import traceback

from loguru import logger

from recipe.function_vectors.prompt_based_function_vector import prompt_function_vector_main

if __name__ == "__main__":
    logger.info(f"args: {sys.argv[1:]}")
    try:
        prompt_function_vector_main(sys.argv[1:])
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
