FROM node:18

WORKDIR /app

COPY package.json .

# Install Node.js dependencies
RUN npm install

COPY . .

EXPOSE 8080

CMD ["node", "user-apps.js"]
