# --- Stage 1: Build ---
FROM node:20-alpine AS build

# 安装 pnpm
RUN npm install -g pnpm

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY package.json pnpm-lock.yaml ./

# 设置 pnpm 使用淘宝源
RUN pnpm config set registry https://registry.npmmirror.com

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制所有前端代码
COPY . .

# 设置 VITE_API_URL 为相对路径，以便通过 Nginx 代理
# 这样可以避免 CORS 问题，因为请求是同源的
ENV VITE_API_URL=/api

# 构建项目
RUN pnpm build

# --- Stage 2: Serve ---
FROM nginx:alpine

# 复制构建好的静态文件
COPY --from=build /app/dist /usr/share/nginx/html

# 复制 Nginx 配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80
