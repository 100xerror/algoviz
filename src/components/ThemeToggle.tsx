import React from 'react';
import { Moon, Sun } from 'lucide-react';

const ThemeToggle = ({ isDarkTheme, toggleTheme }) => {
  return (
    <button
      onClick={toggleTheme}
      className={`p-2 rounded-full transition-colors ${
        isDarkTheme 
          ? 'bg-gray-700 hover:bg-gray-600' 
          : 'bg-gray-200 hover:bg-gray-300'
      }`}
      aria-label={isDarkTheme ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      {isDarkTheme ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  );
};

export default ThemeToggle;