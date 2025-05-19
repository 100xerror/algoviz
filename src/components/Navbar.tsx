import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Code2, ChevronDown } from 'lucide-react';
import { algorithmsData } from '../data/algorithms';

const Navbar = () => {
  const [isOpen, setIsOpen] = React.useState(false);
  const navigate = useNavigate();

  const handleAlgorithmSelect = (id: number) => {
    navigate(`/algorithm/${id}`);
    setIsOpen(false);
  };

  return (
    <nav className="bg-gray-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Code2 className="h-8 w-8 text-blue-500" />
            <span className="ml-2 text-xl font-bold">AlgoVault</span>
          </div>
          
          <div className="relative">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-white"
            >
              <span>Quick Access</span>
              <ChevronDown className="ml-2 h-4 w-4" />
            </button>

            {isOpen && (
              <div className="absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-gray-700 ring-1 ring-black ring-opacity-5 z-50">
                <div className="py-1" role="menu" aria-orientation="vertical">
                  {algorithmsData.map((algo) => (
                    <button
                      key={algo.id}
                      onClick={() => handleAlgorithmSelect(algo.id)}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-100 hover:bg-gray-600"
                      role="menuitem"
                    >
                      {algo.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;