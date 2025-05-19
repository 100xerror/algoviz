import React from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { FileCode2 } from 'lucide-react';
import { algorithmsData } from '../data/algorithms';

const Sidebar = () => {
  const navigate = useNavigate();
  const { id } = useParams();

  return (
    <div className="w-64 bg-gray-800 h-screen flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-sm font-semibold text-gray-400 uppercase">EXPLORER</h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="p-2">
          {algorithmsData.map((algo) => (
            <button
              key={algo.id}
              onClick={() => navigate(`/algorithm/${algo.id}`)}
              className={`w-full flex items-center px-3 py-2 text-sm rounded-md transition-colors ${
                Number(id) === algo.id 
                  ? 'bg-gray-700 text-white' 
                  : 'text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
            >
              <FileCode2 size={16} className="mr-2" />
              {algo.name}.cpp
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;