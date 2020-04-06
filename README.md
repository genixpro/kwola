Kwola
=====

Kwola is an AI user that helps you find bugs in your software. Unlike conventional Selenium or Cypress tests,
Kwola will learn to use your user-interface completely automatically, with no configuration.

Please go to https://kwola.io/ or https://kwolatesting.com/ to learn more.

Installation
============

Dependencies
------------

1) Python

Please go to https://www.python.org/downloads/ to download and install Python.

2) NodeJS

Please go to https://nodejs.org/en/ to install NodeJS or install it through a package manager
appropriate for your distribution.

3) Chromium / Google Chrome

Please go to https://www.chromium.org/ to install Chromium (the open source version of Chrome), or 
go to https://chrome.google.com to install Google Chrome (you may have it installed already).

4) Chromedriver

Install chromedriver globally to your operating system. Go here: https://chromedriver.chromium.org/getting-started
and install the binary appropriate for your operating system and which Chrome version you have installed.

For example, I run the following on Linux with Chrome 80:

`[user@localhost]$ wget https://chromedriver.storage.googleapis.com/80.0.3987.106/chromedriver_linux64.zip`

`[user@localhost]$ unzip chromedriver_linux64.zip`

`[user@localhost]$ sudo cp chromedriver /usr/bin/`

Alternatively you can also install chromedriver just to your Python virtual environment.

5) [optional] Nvidia Drivers & Cuda

How you install NVIDIA Drivers depends on your operating system. You only need to install NVIDIA drivers if you 
are using GPUs. Using GPUs is recommended but optional, and you can do test runs with just the CPU.

Windows: Go here for instructions: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

MacOS: https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html

Ubuntu: https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux
* Its recommended to stick to LTS releases for Ubuntu when doing high performance computing.

Fedora: https://www.if-not-true-then-false.com/2015/fedora-nvidia-guide/

6) [optional] Docker

Go here and follow the instructions to install Docker: https://docs.docker.com/install/

Docker not required for Kwola to operate but is used in some of the examples.

Instructions
------------

Clone the git repository.

`[user@localhost]$ git clone https://github.com/Kwola/kwola.git`

Create your Python virtual environment

`[user@localhost]$ cd kwola`

`[user@localhost]$ python3 -m venv venv`

`[user@localhost]$ source venv/bin/activate`

Install main dependencies.

`[user@localhost]$ npm install`

`[user@localhost]$ python3 setup.py develop`

Install babel-cli globally. This makes it easier for the code to access the babel binary.

`[user@localhost]$ npm install @babel/cli -g`

And that's it! 

Usage
=====

Running Kwola is very straightforward. To initiate a Kwola testing session, run the following command. 
Make sure to replace the URL with the url pointing to the website you want to start testing. The URL
must be a complete, fully validated url containing the http:// part and everything. Also for now, its
easiest just to run it from within the Kwola directory.

`[user@localhost]$ cd kwola`

`[user@localhost]$ kwola http://yoururl.com/`

Kwola will now start testing your application! Its that easy. Kwola will create a directory to hold
all of its results in your current directory. You can cancel the run simply using Ctrl-C or Cmd-C in
your terminal. If you want to restart the run, simply run kwola with no arguments:

`[user@localhost]$ kwola`

Or alternatively, you can run Kwola and give it a specific directory containing a Kwola run. This
allows you to restart specific runs.

`[user@localhost]$ kwola kwola_run_0`

For example, you can use Kwola with one of our example codebases if you have Docker installed:

`[user@localhost]$ docker run docker.io/kwola/kros-1`

`[user@localhost]$ kwola http://172.17.0.2:3000/`

You will then see Kwola running on our restaraunt backend sample codebase.

Support
=======

Kwola is maintained by Kwola Software Testing Inc, a Canadian company. You can contact the authors at any of the following emails: quinn@kwola.io, brad@kwola.io or daniel@kwola.io

Roadmap
=======

Oh boy there is a lot of stuff on the roadmap for Kwola. Too much to write down today.

Contributing
============

We are absolutely open to contributors. If you are interested in Kwola, please reach out to brad@kwola.io via email and we will get in touch. We will accept any contributions so long as the copyrights are transferred to the Kwola Software Testing company.

Or alternatively just throw up a pull-request. That always gets peoples attention ;)

Authors and acknowledgment
==========================

The project was founded by Quinn Lawson, Bradley Arsenault and Daniel Shakmundes.

License
=======

The project is licensed GPLv3 Affero. 

Project status
==============

As of February 2020, the project is just getting started. We are open to anyone who wants to join!

