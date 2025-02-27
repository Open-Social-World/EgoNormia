from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    # Information
    name="egonormia",
    description="Benchmarking Physical Social Norm Understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.0.0",
    url="https://github.com/Open-Social-World/EgoNormia",
    author="MohammadHossein Rezaei, Yicheng Fu, Phil Cuvin",
    license="Apache",
    keywords="vlm ai nlp llm",
    project_urls={
        "website": "",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'tqdm',
        'spacy',
        'google-genai',
        'requests',
        'httpx',
        'Pillow',
        'openai',
        'anthropic',
        'opencv-python',
        'python-dotenv',
        'datasets',
        'decord',
        'faiss-cpu',
        'torch',
        'mkdocs'
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
