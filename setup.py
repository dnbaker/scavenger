import os
from setuptools import setup


def pip_to_requirements(s):
    import re
    """
    Change a PIP-style requirements.txt string into one suitable for setup.py
    """

    if s.startswith('#'):
        return ''
    m = re.match('(.*)([>=]=[.0-9]*).*', s)
    if m:
        return '%s (%s)' % (m.group(1), m.group(2))
    return s.strip()


DESCRIPTION = 'Multi-modal and batch-corrected self-supervised learning for Spatial/Single-Cell experiments combined with biomedical images and biomedical annotations.'
setup(
    name='scavenger',
    version=open('VERSION', 'r').read().strip(),
    author='dnbaker',
    author_email='d.nephi.baker@gmail.com',
    license='MIT',
    url='https://github.com/dnbaker/',

    install_requires=open('requirements.txt').readlines(),
    #extras_require=dict(
    #    dev=open('requirements-dev.txt').readlines()
    #),

    description=DESCRIPTION,
    long_description=open('README.rst', 'r').read() if os.path.isfile("README.rst") else DESCRIPTION,
    keywords=['python', 'single-cell', 'spatial'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    packages=["scavenger"],
)
