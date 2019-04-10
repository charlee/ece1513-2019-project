# ECE1513

## Environment Setup

**Note**: This project requires Python 3.6 (not compatible with Python 3.7).
If you are using macOS with Homebrew, please install Python 3.6 with the following commands
([credit](https://apple.stackexchange.com/questions/329187/homebrew-rollback-from-python-3-7-to-python-3-6-5-x)):

```
$ brew unlink python
$ brew install --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
```
(Note the `--ignore-dependencies` flag - there is an [dependency issue](https://github.com/tensorflow/tensorflow/issues/25093)) which prevents Python 3.6.5 from installing.)

First create a virtualenv in the project folder:

    $ python3 -m venv .env

Activate the virtualenv (this is required each time terminal is reopened):

    $ source .env/bin/activate

For Windows, use instead

    $.env\Scripts\activate.bat

Install necessary packages:

    $ pip install -r requirements.txt
