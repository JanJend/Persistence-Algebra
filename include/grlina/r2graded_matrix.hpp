/**
 * @file r2graded_matrix.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once

#ifndef R2GRADED_MATRIX_HPP
#define R2GRADED_MATRIX_HPP

#include <iostream>
#include <vector>
#include <grlina/graded_matrix.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>



namespace graded_linalg {

using r2degree = std::pair<double, double>;
template<>
struct Degree_traits<r2degree> {
    static bool equals(const r2degree& lhs, const r2degree& rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    static bool smaller(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first < rhs.first) {
            return (lhs.second <= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        } else {
            return false;
        }
    }

    static bool greater(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first > rhs.first) {
            return (lhs.second >= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second > rhs.second;
        } else {
            return false;
        }
    }

    static bool greater_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first >= rhs.first) && (lhs.second >= rhs.second);
    }

    static bool smaller_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first <= rhs.first) && (lhs.second <= rhs.second);
    }

    static bool lex_order(const r2degree& a, const r2degree& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

    static std::function<bool(const r2degree&, const r2degree&)> lex_lambda;

    static vec<double> position(const r2degree& a)  {
        return {a.first, a.second};
    }

    static void print_degree(const r2degree& a) {
        std::cout << "(" << a.first << ", " << a.second << ")";
    }

    static r2degree join(const r2degree& a, const r2degree& b)  {
        return {std::max(a.first, b.first), std::max(a.second, b.second)};
    }

    static r2degree meet(const r2degree& a, const r2degree& b) {
        return {std::min(a.first, b.first), std::min(a.second, b.second)};
    }

    
    /**
     * @brief Writes the r2degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const r2degree& a) {
        os << a.first << " " << a.second;
    }

    template <typename InputStream>
    static r2degree from_stream(InputStream& iss){
        r2degree deg;
        iss >> deg.first >> deg.second;
        return deg;
    }

}; //Degree_traits<r2degree>

/**
 * @brief Lambda function to compare lexicographically for sorting.
 */
std::function<bool(const r2degree&, const r2degree&)> Degree_traits<r2degree>::lex_lambda = [](const r2degree& a, const r2degree& b) {
    return Degree_traits<r2degree>::lex_order(a, b);
};

/**
 * @brief Michael Kerbers Grid_scheduler class from mpfree for fast kernel computation.
 * 

template <typename index>
class Grid_scheduler {

    public:
  
      std::priority_queue< pair<index>, vec<pair<index>>, Sort_grades> grades;
  
      std::map< pair<index> , pair<index> > index_range;
  
      pair<index> curr_grade;
  
      Grid_scheduler() {}
  
      // It is assumed that columns with the same grade appear in M consecutively
      template<typename GradedMatrix>
        Grid_scheduler(GradedMatrix& M) {
  
        //std::cout << "Grid scheduler with matrix having " << M.num_grades_x << " x-grades and " << M.num_grades_y << " y-grades" << std::endl;
        
        index_pair last_pair=std::make_pair(-1,-1);
        index curr_start=-1;
        for(int i=0;i<M.get_num_cols();i++) {
      index curr_x=M.grades[i].index_at[0];
      index curr_y=M.grades[i].index_at[1];
      assert(curr_x<M.num_grades_x);
      assert(curr_y<M.num_grades_y);
      if(curr_x!=last_pair.first || curr_y !=last_pair.second) {
        // New grade
        if(curr_start!=-1) {
          index_range[last_pair]=std::make_pair(curr_start,i);
        }
        curr_start=i;
        last_pair = std::make_pair(curr_x,curr_y);
        grades.push(last_pair);
      }
        }
        if(curr_start!=-1) {
      index_range[last_pair]=std::make_pair(curr_start,M.get_num_cols());
        }
        curr_grade=std::make_pair(-1,-1);
        
      }
  
      int size() {
        return grades.size();
      }
  
      bool at_end() {
        return grades.empty();
      }
  
      index_pair next_grade() {
        index_pair result = grades.top();
        grades.pop();
        while(!grades.empty() && grades.top()==result) {
      grades.pop();
        }
        curr_grade=result;
        return result;
      }
  
      index_pair index_range_at(index x, index y) {
        auto find_grade = index_range.find(std::make_pair(x,y));
        if(find_grade==index_range.end()) {
      return std::make_pair(0,0);
        }
        return find_grade->second;
      }
  
      void notify(index x,index y) {
        //std::cout << "Got notified about " << x << " " << y << std::endl;
        if(curr_grade.first!=x || curr_grade.second!=y) {
      grades.push(std::make_pair(x,y));
        }
      }
      
    };

*/


/**
 * @brief A graded matrix with degrees in R^2.
 * 
 * @tparam index 
 */
template <typename index>
struct R2GradedSparseMatrix : GradedSparseMatrix<r2degree, index> {

    R2GradedSparseMatrix() : GradedSparseMatrix<r2degree, index>() {}
    R2GradedSparseMatrix(index m, index n) : GradedSparseMatrix<r2degree, index>(m, n) {}
    

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) 
        : GradedSparseMatrix<r2degree, index>(filepath, lex_sort, compute_batches) {
    } // Constructor from file


    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<r2degree, index>(file_stream, lex_sort, compute_batches) {
    } // Constructor from ifstream

    

    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r2(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "2" << std::endl;
        output_stream << this->num_cols << " " << this->num_rows << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->num_cols; ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->num_rows; ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
            output_stream << std::endl;
        }
    }

    /**
     * @brief Returns a basis for the kernel of a 2d graded matrix. 
     * Assumes that the columns are sorted lexicographically.
     * 
     * @return SparseMatrix<index> 
     */
    R2GradedSparseMatrix r2kernel() {

    }




}; // R2GradedSparseMatrix




} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP
