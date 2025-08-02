/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        accent: {
          50: '#fef7ff',
          100: '#fceeff',
          200: '#f8daff',
          300: '#f3beff',
          400: '#ea92ff',
          500: '#dc66ff',
          600: '#c241f7',
          700: '#a825d8',
          800: '#8a20b0',
          900: '#721e8f',
        },
        neural: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
        }
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: '#1e293b',
            a: {
              color: '#dc66ff',
              '&:hover': {
                color: '#a825d8',
              },
            },
            'h1, h2, h3, h4': {
              color: '#0f172a',
            },
            code: {
              color: '#dc66ff',
              backgroundColor: '#f8fafc',
              padding: '0.25rem 0.5rem',
              borderRadius: '0.375rem',
              fontWeight: '600',
              border: '1px solid #e2e8f0',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
            pre: {
              backgroundColor: '#0f172a',
              color: '#f8fafc',
              border: '1px solid #334155',
            },
            'pre code': {
              backgroundColor: 'transparent',
              color: 'inherit',
              border: 'none',
            },
            blockquote: {
              borderLeftColor: '#dc66ff',
              backgroundColor: '#fef7ff',
            },
            'blockquote p:first-of-type::before': {
              content: '""',
            },
            'blockquote p:last-of-type::after': {
              content: '""',
            },
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}

