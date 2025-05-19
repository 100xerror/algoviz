import React, { useRef } from 'react';
import { Copy, Check } from 'lucide-react';

const CodeDisplay = ({ selectedAlgorithm, isDarkTheme }) => {
  const [copied, setCopied] = React.useState(false);
  const codeRef = useRef(null);

  const copyToClipboard = () => {
    if (selectedAlgorithm && codeRef.current) {
      navigator.clipboard.writeText(selectedAlgorithm.code).then(
        () => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        },
        (err) => {
          console.error('Could not copy text: ', err);
        }
      );
    }
  };

  if (!selectedAlgorithm) {
    return (
      <div className={`w-full md:w-2/3 lg:w-3/4 ${isDarkTheme ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-lg p-8 flex flex-col justify-center items-center min-h-[50vh] transition-all duration-300`}>
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className={`${isDarkTheme ? 'text-gray-600' : 'text-gray-300'} mb-4`}>
          <polyline points="16 18 22 12 16 6"></polyline>
          <polyline points="8 6 2 12 8 18"></polyline>
        </svg>
        <h3 className="text-xl font-semibold">Select an Algorithm</h3>
        <p className={`mt-2 ${isDarkTheme ? 'text-gray-400' : 'text-gray-500'}`}>
          Click on an algorithm from the list to view its C++ implementation
        </p>
      </div>
    );
  }

  return (
    <div className={`w-full md:w-2/3 lg:w-3/4 ${isDarkTheme ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-lg transition-all duration-300`}>
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="text-xl font-bold">{selectedAlgorithm.name}</h2>
        <button
          onClick={copyToClipboard}
          className={`flex items-center space-x-1 px-3 py-1 rounded-md transition-colors ${
            isDarkTheme 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-200 text-gray-600'
          }`}
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check size={18} className="text-green-500" />
              <span className="text-sm">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={18} />
              <span className="text-sm">Copy</span>
            </>
          )}
        </button>
      </div>
      
      <div 
        className={`p-4 overflow-auto max-h-[70vh] font-mono text-sm whitespace-pre ${
          isDarkTheme ? 'bg-gray-900' : 'bg-gray-50'
        }`}
      >
        <pre ref={codeRef}>{selectedAlgorithm.code}</pre>
      </div>
    </div>
  );
};

export default CodeDisplay;