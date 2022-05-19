from setuptools import setup

if __name__ == "__main__":
    try:
        setup()
    except Exception as error_instance:  # noqa
        print(error_instance)
        raise
