# Running the Flash Attention Visualization Locally

This guide walks you through setting up and running the Flash Attention visualization on your local machine.

## Prerequisites

You need Node.js installed. Check if you have it:

```bash
node --version
```

If not installed, download from [nodejs.org](https://nodejs.org/) (use the LTS version).

## Setup Steps

### 1. Create a new React project with Vite

```bash
npm create vite@latest flash-attention-demo -- --template react
cd flash-attention-demo
npm install
```

### 2. Install Tailwind CSS

The visualization uses Tailwind CSS for styling.

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 3. Configure Tailwind

Edit `tailwind.config.js` to contain:

```js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### 4. Set up Tailwind CSS imports

Replace the contents of `src/index.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 5. Add the visualization component

Save `flash_attention_viz.jsx` as `src/FlashAttentionViz.jsx` in your project.

### 6. Update App.jsx

Replace the contents of `src/App.jsx` with:

```jsx
import FlashAttentionViz from './FlashAttentionViz';

function App() {
  return <FlashAttentionViz />;
}

export default App;
```

### 7. Run the development server

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

## Final Project Structure

```
flash-attention-demo/
├── src/
│   ├── App.jsx                # imports and renders FlashAttentionViz
│   ├── FlashAttentionViz.jsx  # the visualization code
│   ├── index.css              # Tailwind imports
│   └── main.jsx               # entry point (don't modify)
├── tailwind.config.js         # Tailwind configuration
├── postcss.config.js          # PostCSS configuration (auto-generated)
├── package.json
└── index.html
```

## Troubleshooting

**Styles not appearing?**
- Make sure `src/index.css` has the three `@tailwind` directives
- Check that `tailwind.config.js` has the correct `content` paths
- Restart the dev server after config changes

**Component not rendering?**
- Check the browser console for errors
- Verify the import path in `App.jsx` matches the filename exactly (case-sensitive)

**Port 5173 in use?**
- Vite will automatically try the next available port, or run with a specific port:
  ```bash
  npm run dev -- --port 3000
  ```
