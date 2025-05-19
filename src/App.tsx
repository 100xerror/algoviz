import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import AlgorithmPage from './components/AlgorithmPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white flex">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navbar />
          <Routes>
            <Route path="/" element={
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <h1 className="text-4xl font-bold mb-4">Welcome to AlgoVault</h1>
                  <p className="text-gray-400">Select an algorithm from the sidebar to view its implementation</p>
                </div>
              </div>
            } />
            <Route path="/algorithm/:id" element={<AlgorithmPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App