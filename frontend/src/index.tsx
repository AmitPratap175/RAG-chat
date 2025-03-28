import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // Ensure this file exists or update the path if necessary

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const TemplateWebsite: React.FC = () => (
  <div>
    <header>
      <h1>Welcome to Our Website</h1>
    </header>
    <main>
      <p>This is a template website with some content.</p>
      <p>Feel free to explore and interact with the customer support chat.</p>
    </main>
    <App />
  </div>
);

root.render(<TemplateWebsite />);