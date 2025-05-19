import React from 'react';

const AlgorithmList = ({ algorithms, selectedAlgorithm, onSelectAlgorithm, isDarkTheme }) => {
  return (
    <div className={`w-full md:w-1/3 lg:w-1/4 ${isDarkTheme ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-lg p-4 transition-all duration-300`}>
      <h2 className="text-xl font-bold mb-4 border-b pb-2 border-gray-700">Algorithms</h2>
      <div className="flex flex-col space-y-2">
        {algorithms.map((algorithm) => (
          <button
            key={algorithm.id}
            onClick={() => onSelectAlgorithm(algorithm)}
            className={`p-3 rounded-md text-left transition-all duration-200 flex items-center
              ${selectedAlgorithm?.id === algorithm.id 
                ? (isDarkTheme ? 'bg-blue-600 text-white' : 'bg-blue-500 text-white') 
                : (isDarkTheme ? 'hover:bg-gray-700' : 'hover:bg-gray-200')}`
              }
          >
            <span className="w-8 h-8 rounded-full flex items-center justify-center mr-3 bg-opacity-20 bg-blue-500">
              {algorithm.id}
            </span>
            <span>{algorithm.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default AlgorithmList;