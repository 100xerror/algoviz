export const algorithmsData = [
  {
    id: 1,
    name: "N-Queens",
    code: `#include <iostream>
#include <vector>
using namespace std;

class NQueens {
private:
    int n;
    vector<vector<string>> solutions;
    
    bool isSafe(vector<string>& board, int row, int col) {
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        
        // Check upper-left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        // Check upper-right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        return true;
    }
    
    void solve(vector<string>& board, int row) {
        if (row == n) {
            solutions.push_back(board);
            return;
        }
        
        for (int col = 0; col < n; col++) {
            if (isSafe(board, row, col)) {
                board[row][col] = 'Q';
                solve(board, row + 1);
                board[row][col] = '.'; // Backtrack
            }
        }
    }
    
public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        vector<string> board(n, string(n, '.'));
        solve(board, 0);
        return solutions;
    }
    
    void printSolution() {
        for (const auto& solution : solutions) {
            for (const auto& row : solution) {
                cout << row << endl;
            }
            cout << endl;
        }
        cout << "Total solutions: " << solutions.size() << endl;
    }
};

int main() {
    int n = 4; // Change this to solve for different board sizes
    NQueens solver;
    solver.solveNQueens(n);
    solver.printSolution();
    return 0;
}
`
  },
  {
    id: 2,
    name: "Sudoku Solver",
    code: `#include <iostream>
#include <vector>
using namespace std;

class SudokuSolver {
private:
    bool isValid(vector<vector<char>>& board, int row, int col, char num) {
        // Check row
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == num) return false;
        }
        
        // Check column
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == num) return false;
        }
        
        // Check 3x3 box
        int startRow = 3 * (row / 3);
        int startCol = 3 * (col / 3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[startRow + i][startCol + j] == num) return false;
            }
        }
        
        return true;
    }
    
    bool solve(vector<vector<char>>& board) {
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                if (board[row][col] == '.') {
                    for (char num = '1'; num <= '9'; num++) {
                        if (isValid(board, row, col, num)) {
                            board[row][col] = num;
                            
                            if (solve(board)) {
                                return true;
                            }
                            
                            board[row][col] = '.'; // Backtrack
                        }
                    }
                    return false; // If no valid number found
                }
            }
        }
        return true; // All cells filled
    }
    
public:
    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
    }
    
    void printBoard(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                cout << board[i][j] << " ";
                if (j == 2 || j == 5) cout << "| ";
            }
            cout << endl;
            if (i == 2 || i == 5) {
                cout << "---------------------" << endl;
            }
        }
    }
};

int main() {
    vector<vector<char>> board = {
        {'5','3','.','.','7','.','.','.','.'},
        {'6','.','.','1','9','5','.','.','.'},
        {'.','9','8','.','.','.','.','6','.'},
        {'8','.','.','.','6','.','.','.','3'},
        {'4','.','.','8','.','3','.','.','1'},
        {'7','.','.','.','2','.','.','.','6'},
        {'.','6','.','.','.','.','2','8','.'},
        {'.','.','.','4','1','9','.','.','5'},
        {'.','.','.','.','8','.','.','7','9'}
    };
    
    SudokuSolver solver;
    cout << "Before solving:" << endl;
    solver.printBoard(board);
    
    solver.solveSudoku(board);
    
    cout << "\\nAfter solving:" << endl;
    solver.printBoard(board);
    
    return 0;
}
`
  },
  {
    id: 3,
    name: "Graph Coloring",
    code: `#include <iostream>
#include <vector>
using namespace std;

class GraphColoring {
private:
    int V; // Number of vertices
    vector<vector<int>> graph;
    vector<int> colors;
    
    bool isSafe(int vertex, int color) {
        for (int i = 0; i < V; i++) {
            if (graph[vertex][i] && colors[i] == color) {
                return false;
            }
        }
        return true;
    }
    
    bool colorGraphUtil(int vertex, int numColors) {
        // Base case: If all vertices are colored
        if (vertex == V) {
            return true;
        }
        
        // Try different colors for vertex
        for (int color = 1; color <= numColors; color++) {
            if (isSafe(vertex, color)) {
                colors[vertex] = color;
                
                // Recur to assign colors to rest of the vertices
                if (colorGraphUtil(vertex + 1, numColors)) {
                    return true;
                }
                
                // If assigning color doesn't lead to a solution, backtrack
                colors[vertex] = 0;
            }
        }
        
        return false; // No solution exists
    }
    
public:
    GraphColoring(int vertices) : V(vertices) {
        graph.resize(V, vector<int>(V, 0));
        colors.resize(V, 0);
    }
    
    void addEdge(int v, int w) {
        graph[v][w] = 1;
        graph[w][v] = 1; // For undirected graph
    }
    
    bool colorGraph(int numColors) {
        if (!colorGraphUtil(0, numColors)) {
            cout << "Solution does not exist" << endl;
            return false;
        }
        
        printSolution();
        return true;
    }
    
    void printSolution() {
        cout << "Solution exists: Following are the assigned colors" << endl;
        for (int i = 0; i < V; i++) {
            cout << "Vertex " << i << " --> Color " << colors[i] << endl;
        }
    }
};

int main() {
    GraphColoring g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(0, 3);
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    
    int numColors = 3;
    g.colorGraph(numColors);
    
    return 0;
}
`
  },
  {
    id: 4,
    name: "Longest Common Subsequence",
    code: `#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class LCS {
public:
    // Bottom-up DP approach
    int findLCS(string text1, string text2) {
        int m = text1.length();
        int n = text2.length();
        
        // Create a DP table
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        // Fill the dp table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        // Reconstruct the LCS
        string lcs = "";
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (text1[i - 1] == text2[j - 1]) {
                lcs = text1[i - 1] + lcs;
                i--; j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        
        cout << "Longest Common Subsequence: " << lcs << endl;
        return dp[m][n];
    }
    
    // Recursive approach with memoization
    int findLCSMemoization(string text1, string text2) {
        int m = text1.length();
        int n = text2.length();
        vector<vector<int>> memo(m + 1, vector<int>(n + 1, -1));
        return lcsRecursive(text1, text2, m, n, memo);
    }
    
    int lcsRecursive(string& text1, string& text2, int m, int n, vector<vector<int>>& memo) {
        if (m == 0 || n == 0) {
            return 0;
        }
        
        if (memo[m][n] != -1) {
            return memo[m][n];
        }
        
        if (text1[m - 1] == text2[n - 1]) {
            memo[m][n] = 1 + lcsRecursive(text1, text2, m - 1, n - 1, memo);
        } else {
            memo[m][n] = max(
                lcsRecursive(text1, text2, m - 1, n, memo),
                lcsRecursive(text1, text2, m, n - 1, memo)
            );
        }
        
        return memo[m][n];
    }
};

int main() {
    LCS lcs;
    string str1 = "ABCDGH";
    string str2 = "AEDFHR";
    
    cout << "String 1: " << str1 << endl;
    cout << "String 2: " << str2 << endl;
    
    int length = lcs.findLCS(str1, str2);
    cout << "Length of LCS: " << length << endl;
    
    return 0;
}
`
  },
  {
    id: 5,
    name: "0/1 Knapsack",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Knapsack01 {
public:
    // Bottom-up DP approach
    int solveKnapsack(vector<int>& weights, vector<int>& values, int capacity) {
        int n = weights.size();
        vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    // Include the item or exclude it
                    dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    // Can't include the item, so exclude it
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        // Find the items that were included
        vector<int> selected;
        int w = capacity;
        for (int i = n; i > 0; i--) {
            if (dp[i][w] != dp[i - 1][w]) {
                selected.push_back(i - 1);
                w -= weights[i - 1];
            }
        }
        
        cout << "Selected items (index): ";
        for (int i = selected.size() - 1; i >= 0; i--) {
            cout << selected[i] << " ";
        }
        cout << endl;
        
        return dp[n][capacity];
    }
    
    // Recursive approach with memoization
    int knapsackRecursive(vector<int>& weights, vector<int>& values, int capacity) {
        int n = weights.size();
        vector<vector<int>> memo(n + 1, vector<int>(capacity + 1, -1));
        return knapsackMemo(weights, values, capacity, n, memo);
    }
    
    int knapsackMemo(vector<int>& weights, vector<int>& values, int capacity, int n, vector<vector<int>>& memo) {
        if (n == 0 || capacity == 0) {
            return 0;
        }
        
        if (memo[n][capacity] != -1) {
            return memo[n][capacity];
        }
        
        if (weights[n - 1] <= capacity) {
            memo[n][capacity] = max(
                values[n - 1] + knapsackMemo(weights, values, capacity - weights[n - 1], n - 1, memo),
                knapsackMemo(weights, values, capacity, n - 1, memo)
            );
        } else {
            memo[n][capacity] = knapsackMemo(weights, values, capacity, n - 1, memo);
        }
        
        return memo[n][capacity];
    }
};

int main() {
    vector<int> values = {60, 100, 120};
    vector<int> weights = {10, 20, 30};
    int capacity = 50;
    
    cout << "Items (value, weight):" << endl;
    for (int i = 0; i < values.size(); i++) {
        cout << "Item " << i << ": (" << values[i] << ", " << weights[i] << ")" << endl;
    }
    cout << "Knapsack capacity: " << capacity << endl;
    
    Knapsack01 knapsack;
    int maxValue = knapsack.solveKnapsack(weights, values, capacity);
    
    cout << "Maximum value: " << maxValue << endl;
    
    return 0;
}
`
  },
  {
    id: 6,
    name: "Longest Palindromic Subsequence",
    code: `#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class LPS {
public:
    // Bottom-up DP approach
    int findLPS(string s) {
        int n = s.length();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        // All substrings of length 1 are palindromes
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        
        // Fill the dp table
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                
                if (s[i] == s[j] && len == 2) {
                    dp[i][j] = 2;
                } else if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        
        // Reconstruct the palindromic subsequence
        string palindrome = "";
        int i = 0, j = n - 1;
        reconstructPalindrome(s, dp, i, j, palindrome);
        
        cout << "Longest Palindromic Subsequence: " << palindrome << endl;
        return dp[0][n - 1];
    }
    
    void reconstructPalindrome(string& s, vector<vector<int>>& dp, int i, int j, string& palindrome) {
        if (i > j) {
            return;
        }
        
        if (i == j) {
            palindrome += s[i];
            return;
        }
        
        if (s[i] == s[j]) {
            palindrome += s[i];
            reconstructPalindrome(s, dp, i + 1, j - 1, palindrome);
            palindrome += s[j];
        } else if (dp[i + 1][j] > dp[i][j - 1]) {
            reconstructPalindrome(s, dp, i + 1, j, palindrome);
        } else {
            reconstructPalindrome(s, dp, i, j - 1, palindrome);
        }
    }
    
    // Using LCS approach
    int findLPSUsingLCS(string s) {
        string reversed = s;
        reverse(reversed.begin(), reversed.end());
        
        // LPS is the LCS of string and its reverse
        return findLCS(s, reversed);
    }
    
    int findLCS(string text1, string text2) {
        int m = text1.length();
        int n = text2.length();
        
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
};

int main() {
    LPS lps;
    string str = "BBABCBCAB";
    
    cout << "String: " << str << endl;
    
    int length = lps.findLPS(str);
    cout << "Length of Longest Palindromic Subsequence: " << length << endl;
    
    return 0;
}
`
  },
  {
    id: 7,
    name: "Maximal Square",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class MaximalSquare {
public:
    // Bottom-up DP approach
    int findMaximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return 0;
        }
        
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<vector<int>> dp(rows + 1, vector<int>(cols + 1, 0));
        
        int maxSquareLen = 0;
        
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                    maxSquareLen = max(maxSquareLen, dp[i][j]);
                }
            }
        }
        
        return maxSquareLen * maxSquareLen; // Area of the maximal square
    }
    
    // Space-optimized bottom-up DP approach
    int findMaximalSquareOptimized(vector<vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return 0;
        }
        
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<int> dp(cols + 1, 0);
        
        int maxSquareLen = 0;
        int prev = 0; // dp[i-1][j-1]
        
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                int temp = dp[j];
                if (matrix[i - 1][j - 1] == '1') {
                    dp[j] = min({dp[j], dp[j - 1], prev}) + 1;
                    maxSquareLen = max(maxSquareLen, dp[j]);
                } else {
                    dp[j] = 0;
                }
                prev = temp;
            }
        }
        
        return maxSquareLen * maxSquareLen;
    }
    
    void printMatrix(vector<vector<char>>& matrix) {
        for (const auto& row : matrix) {
            for (char c : row) {
                cout << c << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    vector<vector<char>> matrix = {
        {'1', '0', '1', '0', '0'},
        {'1', '0', '1', '1', '1'},
        {'1', '1', '1', '1', '1'},
        {'1', '0', '0', '1', '0'}
    };
    
    MaximalSquare ms;
    cout << "Matrix:" << endl;
    ms.printMatrix(matrix);
    
    int maxArea = ms.findMaximalSquare(matrix);
    cout << "Area of the maximal square: " << maxArea << endl;
    
    return 0;
}
`
  },
  {
    id: 8,
    name: "Activity Selection",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Activity {
    int start, finish;
    int index;
    
    Activity(int s, int f, int i) : start(s), finish(f), index(i) {}
};

class ActivitySelection {
public:
    vector<Activity> selectActivities(vector<Activity>& activities) {
        // Sort activities by finish time
        sort(activities.begin(), activities.end(), 
             [](const Activity& a, const Activity& b) {
                 return a.finish < b.finish;
             });
        
        vector<Activity> selected;
        
        // Select the first activity
        if (!activities.empty()) {
            selected.push_back(activities[0]);
        }
        
        // Consider remaining activities
        int n = activities.size();
        int lastSelected = 0;
        
        for (int i = 1; i < n; i++) {
            // If this activity's start time is >= the finish time of the last selected activity
            if (activities[i].start >= activities[lastSelected].finish) {
                selected.push_back(activities[i]);
                lastSelected = i;
            }
        }
        
        return selected;
    }
    
    void printActivities(const vector<Activity>& activities) {
        cout << "Selected Activities:" << endl;
        cout << "Index\\tStart\\tFinish" << endl;
        for (const auto& activity : activities) {
            cout << activity.index << "\\t" << activity.start << "\\t" << activity.finish << endl;
        }
    }
};

int main() {
    vector<Activity> activities = {
        Activity(1, 2, 0),
        Activity(3, 4, 1),
        Activity(0, 6, 2),
        Activity(5, 7, 3),
        Activity(8, 9, 4),
        Activity(5, 9, 5)
    };
    
    cout << "All Activities:" << endl;
    cout << "Index\\tStart\\tFinish" << endl;
    for (const auto& activity : activities) {
        cout << activity.index << "\\t" << activity.start << "\\t" << activity.finish << endl;
    }
    
    ActivitySelection as;
    vector<Activity> selected = as.selectActivities(activities);
    
    cout << endl;
    as.printActivities(selected);
    cout << "Maximum number of activities: " << selected.size() << endl;
    
    return 0;
}
`
  },
  {
    id: 9,
    name: "Minimum Platforms",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class MinimumPlatforms {
public:
    int findMinPlatforms(vector<int>& arrival, vector<int>& departure) {
        int n = arrival.size();
        
        // Sort arrival and departure times
        sort(arrival.begin(), arrival.end());
        sort(departure.begin(), departure.end());
        
        int platformsNeeded = 1; // At least one platform is needed
        int maxPlatforms = 1;
        
        // Use two pointers to traverse arrival and departure arrays
        int i = 1, j = 0;
        
        while (i < n && j < n) {
            // If a train arrives before the previous train departs
            if (arrival[i] <= departure[j]) {
                platformsNeeded++; // Need one more platform
                i++;
            } else { // A train departs
                platformsNeeded--; // One platform becomes free
                j++;
            }
            
            maxPlatforms = max(maxPlatforms, platformsNeeded);
        }
        
        return maxPlatforms;
    }
    
    // Method to convert time in "HH:MM" format to minutes since midnight
    void convertToMinutes(vector<string>& times, vector<int>& minutes) {
        for (const string& time : times) {
            int h = stoi(time.substr(0, 2));
            int m = stoi(time.substr(3, 2));
            minutes.push_back(h * 60 + m);
        }
    }
};

int main() {
    // Times in 24-hour format
    vector<int> arrival = {900, 940, 950, 1100, 1500, 1800};
    vector<int> departure = {910, 1200, 1120, 1130, 1900, 2000};
    
    cout << "Train Schedules:" << endl;
    cout << "Arrival\\tDeparture" << endl;
    for (int i = 0; i < arrival.size(); i++) {
        cout << arrival[i] << "\\t" << departure[i] << endl;
    }
    
    MinimumPlatforms mp;
    int minPlatforms = mp.findMinPlatforms(arrival, departure);
    
    cout << "\\nMinimum platforms required: " << minPlatforms << endl;
    
    // Example with time strings
    vector<string> arrivalTimes = {"09:00", "09:40", "09:50", "11:00", "15:00", "18:00"};
    vector<string> departureTimes = {"09:10", "12:00", "11:20", "11:30", "19:00", "20:00"};
    
    vector<int> arrivalMinutes, departureMinutes;
    mp.convertToMinutes(arrivalTimes, arrivalMinutes);
    mp.convertToMinutes(departureTimes, departureMinutes);
    
    int minPlatformsFromStrings = mp.findMinPlatforms(arrivalMinutes, departureMinutes);
    cout << "Minimum platforms required (using string times): " << minPlatformsFromStrings << endl;
    
    return 0;
}
`
  },
  {
    id: 10,
    name: "Job Scheduling with Deadlines",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Job {
    char id;
    int deadline;
    int profit;
    
    Job(char i, int d, int p) : id(i), deadline(d), profit(p) {}
};

class JobSchedulingWithDeadlines {
public:
    vector<Job> scheduleJobs(vector<Job>& jobs) {
        // Sort jobs by profit in descending order
        sort(jobs.begin(), jobs.end(), 
             [](const Job& a, const Job& b) {
                 return a.profit > b.profit;
             });
        
        // Find the maximum deadline
        int maxDeadline = 0;
        for (const auto& job : jobs) {
            maxDeadline = max(maxDeadline, job.deadline);
        }
        
        // Initialize the result sequence with 'false' entries
        vector<bool> slot(maxDeadline, false);
        vector<Job> result;
        
        // Process jobs
        for (const auto& job : jobs) {
            // Find a free slot for this job
            for (int i = min(maxDeadline - 1, job.deadline - 1); i >= 0; i--) {
                if (!slot[i]) {
                    result.push_back(job);
                    slot[i] = true;
                    break;
                }
            }
        }
        
        return result;
    }
    
    void printJobs(const vector<Job>& jobs) {
        cout << "Scheduled Jobs:" << endl;
        cout << "ID\\tDeadline\\tProfit" << endl;
        for (const auto& job : jobs) {
            cout << job.id << "\\t" << job.deadline << "\\t\\t" << job.profit << endl;
        }
    }
    
    int getTotalProfit(const vector<Job>& jobs) {
        int totalProfit = 0;
        for (const auto& job : jobs) {
            totalProfit += job.profit;
        }
        return totalProfit;
    }
};

int main() {
    vector<Job> jobs = {
        Job('a', 2, 100),
        Job('b', 1, 19),
        Job('c', 2, 27),
        Job('d', 1, 25),
        Job('e', 3, 15)
    };
    
    cout << "Available Jobs:" << endl;
    cout << "ID\\tDeadline\\tProfit" << endl;
    for (const auto& job : jobs) {
        cout << job.id << "\\t" << job.deadline << "\\t\\t" << job.profit << endl;
    }
    
    JobSchedulingWithDeadlines js;
    vector<Job> scheduled = js.scheduleJobs(jobs);
    
    cout << endl;
    js.printJobs(scheduled);
    
    int totalProfit = js.getTotalProfit(scheduled);
    cout << "\\nTotal profit: " << totalProfit << endl;
    
    return 0;
}
`
  },
  {
    id: 11,
    name: "Fractional Knapsack",
    code: `#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Item {
    int weight;
    int value;
    double valuePerWeight;
    int index;
    
    Item(int w, int v, int i) : weight(w), value(v), index(i) {
        valuePerWeight = (double)value / weight;
    }
};

class FractionalKnapsack {
public:
    double getMaxValue(vector<Item>& items, int capacity) {
        // Sort items by value per unit weight in descending order
        sort(items.begin(), items.end(), 
             [](const Item& a, const Item& b) {
                 return a.valuePerWeight > b.valuePerWeight;
             });
        
        double totalValue = 0.0;
        int currentWeight = 0;
        
        cout << "Items selected:" << endl;
        cout << "Index\\tWeight\\tValue\\tFraction" << endl;
        
        for (const auto& item : items) {
            if (currentWeight + item.weight <= capacity) {
                // Take the whole item
                currentWeight += item.weight;
                totalValue += item.value;
                cout << item.index << "\\t" << item.weight << "\\t" << item.value << "\\t" << 1.0 << endl;
            } else {
                // Take a fraction of the item
                int remainingCapacity = capacity - currentWeight;
                double fraction = (double)remainingCapacity / item.weight;
                totalValue += item.value * fraction;
                cout << item.index << "\\t" << item.weight << "\\t" << item.value << "\\t" << fraction << endl;
                break;
            }
        }
        
        return totalValue;
    }
};

int main() {
    vector<Item> items = {
        Item(10, 60, 0),
        Item(20, 100, 1),
        Item(30, 120, 2)
    };
    
    int capacity = 50;
    
    cout << "Available Items:" << endl;
    cout << "Index\\tWeight\\tValue\\tValue/Weight" << endl;
    for (const auto& item : items) {
        cout << item.index << "\\t" << item.weight << "\\t" << item.value << "\\t" << item.valuePerWeight << endl;
    }
    
    cout << "\\nKnapsack capacity: " << capacity << endl;
    
    FractionalKnapsack knapsack;
    double maxValue = knapsack.getMaxValue(items, capacity);
    
    cout << "\\nMaximum value: " << maxValue << endl;
    
    return 0;
}
`
  },
  {
    id: 12,
    name: "Huffman Coding",
    code: `#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
using namespace std;

struct HuffmanNode {
    char data;
    int freq;
    HuffmanNode *left, *right;
    
    HuffmanNode(char data, int freq) : data(data), freq(freq), left(nullptr), right(nullptr) {}
};

struct CompareNodes {
    bool operator()(HuffmanNode* lhs, HuffmanNode* rhs) {
        return lhs->freq > rhs->freq;
    }
};

class HuffmanCoding {
private:
    void generateCodes(HuffmanNode* root, string code, unordered_map<char, string>& huffmanCode) {
        if (!root) {
            return;
        }
        
        // If this is a leaf node, store the code
        if (!root->left && !root->right) {
            huffmanCode[root->data] = (code.empty() ? "1" : code);
        }
        
        // Traverse left subtree
        generateCodes(root->left, code + "0", huffmanCode);
        
        // Traverse right subtree
        generateCodes(root->right, code + "1", huffmanCode);
    }
    
    void printTree(HuffmanNode* root, string indent = "") {
        if (!root) {
            return;
        }
        
        // Print right subtree
        printTree(root->right, indent + "    ");
        
        // Print current node
        cout << indent;
        if (root->left || root->right) {
            cout << "+" << root->freq << endl;
        } else {
            cout << "'" << root->data << "' (" << root->freq << ")" << endl;
        }
        
        // Print left subtree
        printTree(root->left, indent + "    ");
    }
    
public:
    unordered_map<char, string> buildHuffmanTree(string text) {
        // Count frequency of each character
        unordered_map<char, int> freq;
        for (char c : text) {
            freq[c]++;
        }
        
        // Create a priority queue to store nodes based on their frequency
        priority_queue<HuffmanNode*, vector<HuffmanNode*>, CompareNodes> pq;
        
        // Create leaf nodes for each character and add them to the priority queue
        for (auto& pair : freq) {
            pq.push(new HuffmanNode(pair.first, pair.second));
        }
        
        // Build Huffman Tree: combine the two nodes of the lowest frequency
        while (pq.size() > 1) {
            // Extract the two nodes with lowest frequency
            HuffmanNode* left = pq.top();
            pq.pop();
            
            HuffmanNode* right = pq.top();
            pq.pop();
            
            // Create a new internal node with these two nodes as children
            // and with frequency equal to the sum of frequencies
            HuffmanNode* newNode = new HuffmanNode('$', left->freq + right->freq);
            newNode->left = left;
            newNode->right = right;
            
            // Add the new node to the priority queue
            pq.push(newNode);
        }
        
        // Get the root of the Huffman Tree
        HuffmanNode* root = pq.top();
        
        // Print the Huffman Tree
        cout << "Huffman Tree:" << endl;
        printTree(root);
        
        // Generate Huffman codes
        unordered_map<char, string> huffmanCode;
        generateCodes(root, "", huffmanCode);
        
        return huffmanCode;
    }
    
    string encodeText(string text, unordered_map<char, string>& huffmanCode) {
        string encodedText = "";
        for (char c : text) {
            encodedText += huffmanCode[c];
        }
        return encodedText;
    }
    
    string decodeText(string encodedText, HuffmanNode* root) {
        string decodedText = "";
        HuffmanNode* current = root;
        
        for (char bit : encodedText) {
            if (bit == '0') {
                current = current->left;
            } else {
                current = current->right;
            }
            
            // If a leaf node is reached
            if (!current->left && !current->right) {
                decodedText += current->data;
                current = root;
            }
        }
        
        return decodedText;
    }
};

int main() {
    string text = "this is an example for huffman encoding";
    
    cout << "Original text: " << text << endl;
    
    HuffmanCoding huffman;
    unordered_map<char, string> huffmanCode = huffman.buildHuffmanTree(text);
    
    // Print the Huffman codes
    cout << "\\nHuffman Codes:" << endl;
    for (auto& pair : huffmanCode) {
        cout << "'" << pair.first << "': " << pair.second << endl;
    }
    
    // Encode the text
    string encodedText = huffman.encodeText(text, huffmanCode);
    cout << "\\nEncoded text: " << encodedText << endl;
    
    // Calculate compression ratio
    int originalSize = text.length() * 8; // Assuming 8 bits per character
    int compressedSize = encodedText.length();
    double ratio = (double)originalSize / compressedSize;
    
    cout << "\\nOriginal size: " << originalSize << " bits" << endl;
    cout << "Compressed size: " << compressedSize << " bits" << endl;
    cout << "Compression ratio: " << ratio << endl;
    
    return 0;
}
`
  }
];