FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	build-essential \
	python-opengl \
	tk-dev \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip setuptools \
	&& pip3 install -U \
	coverage \
	tensorflow \
	tensorflow_probability \
	twine \
	sphinx \
	sphinx-automodapi \
	sphinx_rtd_theme \
	unittest-xml-reporting \
	wheel

CMD ["/bin/bash"]
