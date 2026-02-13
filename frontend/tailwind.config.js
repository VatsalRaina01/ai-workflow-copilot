/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#09090b",
        surface: "#18181b",
        primary: "#3b82f6",
        secondary: "#8b5cf6", 
        accent: "#f472b6",
        text: "#e4e4e7",
        muted: "#a1a1aa",
        border: "#27272a"
      }
    },
  },
  plugins: [],
}
