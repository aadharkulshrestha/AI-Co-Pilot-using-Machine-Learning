/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/copilot/:path*',
        destination: 'http://127.0.0.1:8002/api/copilot/:path*',
      },
      {
        source: '/api/manuals/:path*',
        destination: 'http://127.0.0.1:8002/api/manuals/:path*',
      },
      {
        source: '/api/svs/:path*',
        destination: 'http://127.0.0.1:8002/api/svs/:path*',
      },
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8001/api/:path*',
      },
    ];
  },
};

export default nextConfig;
