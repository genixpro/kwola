# Frozen Kros 1 app source is supplied as the build context.  This file only
# modernizes its container plumbing; it does not alter application code.
FROM node:6.17.1-stretch@sha256:e133e66ec3bfc98da0440e552f452e5cdf6413319d27a2db3b01ac4b319759b3

# Stretch archive signatures are intentionally expired; this frozen,
# isolated compatibility image accepts those archived package records.
RUN sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g; s|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list \
    && sed -i '/stretch-updates/d' /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends python build-essential ruby-sass git netcat-openbsd \
    && npm install --global bower@1.8.14 grunt-cli@1.3.2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json bower.json .bowerrc ./
COPY client ./client
RUN npm install --unsafe-perm && bower install --allow-root
COPY . .
EXPOSE 3000
CMD ["grunt", "serve"]
