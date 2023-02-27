from setuptools import setup, find_packages

setup(
    name='winit',
    version='0.1.0',
    packages=['winit', 'winit.datagen', 'winit.explainer', 'winit.explainer.generator',
              'winit.explainer.attribution', 'winit.explainer.dynamaskutils'],
    url='',
    license='',
    author='Layer 6 AI',
    author_email='kk@layer6.ai',
    description='Experiments regarding the WinIT algorithm',
    python_requires='>=3.8.11',
    install_requires=[
        "numpy==1.22.3",
        "pandas==1.3.5",
        "torch==1.11.0",
        "matplotlib==3.6.0",
        "scikit-learn==1.1.2",
        "scipy==1.6.2",
        "captum==0.4.0",
        "seaborn==0.12.1",
        "timesynth==0.2.4",
    ],
)
