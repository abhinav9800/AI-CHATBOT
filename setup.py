from setuptools import find_packages, setup

setup(
    name="RAG_FUNCTIONS",
    version="0.0.1",
    author="abc",
    author_email="abc@gmail.com",
    packages=find_packages(),
    install_requires=[
    "flask",
    "langchain-community",
    "langchain-chroma",
    "pypdf",
    "langchainhub",
    "transformers",
    "langchain-google-genai"
]


)
    