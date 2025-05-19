import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Copy, Check, ArrowLeft, ArrowRight } from 'lucide-react';
import { algorithmsData } from '../data/algorithms';

const AlgorithmPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [copied, setCopied] = React.useState(false);
  
  const currentIndex = algorithmsData.findIndex(algo => algo.id === Number(id));
  const algorithm = algorithmsData[currentIndex];
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(algorithm.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const navigateToAlgorithm = (direction: 'prev' | 'next') => {
    const newIndex = direction === 'prev' ? currentIndex - 1 : currentIndex + 1;
    if (newIndex >= 0 && newIndex < algorithmsData.length) {
      navigate(`/algorithm/${algorithmsData[newIndex].id}`);
    }
  };

  if (!algorithm) {
    return <div>Algorithm not found</div>;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="bg-gray-800 rounded-lg shadow-xl overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigateToAlgorithm('prev')}
              disabled={currentIndex === 0}
              className={`p-2 rounded-full ${currentIndex === 0 ? 'text-gray-500' : 'hover:bg-gray-700'}`}
            >
              <ArrowLeft size={20} />
            </button>
            <h1 className="text-2xl font-bold">{algorithm.name}</h1>
            <button
              onClick={() => navigateToAlgorithm('next')}
              disabled={currentIndex === algorithmsData.length - 1}
              className={`p-2 rounded-full ${currentIndex === algorithmsData.length - 1 ? 'text-gray-500' : 'hover:bg-gray-700'}`}
            >
              <ArrowRight size={20} />
            </button>
          </div>
          
          <button
            onClick={handleCopy}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md transition-colors"
          >
            {copied ? (
              <>
                <Check size={18} />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy size={18} />
                <span>Copy Code</span>
              </>
            )}
          </button>
        </div>
        
        <div className="overflow-auto max-h-[calc(100vh-12rem)]">
          <pre className="p-4 font-mono text-sm leading-relaxed whitespace-pre">
            <code>{algorithm.code}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmPage;