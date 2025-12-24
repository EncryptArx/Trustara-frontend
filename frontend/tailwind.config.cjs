module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        'neon-cyan': '#00F0FF',
        'neon-magenta': '#FF2AC6',
        'neon-violet': '#9B7CFF',
        'neon-red': '#FF3B3B',
        'neon-green': '#2AFF7E',
        'bg-deep': '#0b0c0f'
      },
      fontFamily: {
        sans: ['Inter', 'Poppins', 'ui-sans-serif', 'system-ui']
      },
      boxShadow: {
        'neon': '0 6px 30px rgba(0,240,255,0.08), 0 0 24px rgba(155,124,255,0.06)'
      }
    }
  },
  plugins: [],
}
