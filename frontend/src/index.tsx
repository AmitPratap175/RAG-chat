import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // Ensure this file exists or update the path if necessary

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const TemplateWebsite: React.FC = () => (
  <div>
    <App />
  </div>
);

root.render(<TemplateWebsite />);