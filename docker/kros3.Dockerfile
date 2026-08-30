# Frozen Kros 3 app source is supplied as the build context.  Node 14 remains
# compatible with the original react-scripts 3 build while avoiding the EOL
# Python 3.6 base image used by the archival Dockerfile.
FROM node:14.21.3-bullseye@sha256:c0bff0d29a742f40650d5f0305dd581351c10954e6cb6676fc96f47590b9666e AS build
WORKDIR /kros3
COPY package.json package-lock.json ./
# The original repository already carries its npm v1 lockfile.  npm ci keeps
# the frozen React application reproducible without changing its sources.
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build

FROM nginx:1.22.1@sha256:fc5f5fb7574755c306aaf88456ebfbe0b006420a184d52b923d2f0197108f6b7
# Preserve the frozen app's original nginx_config, which serves `/build`.
COPY --from=build /kros3/build /build
COPY nginx_config /etc/nginx/conf.d/default.conf
EXPOSE 80
