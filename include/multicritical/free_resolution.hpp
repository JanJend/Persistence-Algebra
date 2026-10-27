#pragma once

// Consolidated upstream conversion with grlina coordinate and upset degrees.

#include <grlina/matrices/coord.hpp>
#include <grlina/matrices/upset.hpp>
#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

#if MULTI_CRITICAL_TIMERS
// BEGIN copied from multicritical/include/multi_critical/boost_timers.h
/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/


#include <iostream>
#include <boost/timer/timer.hpp>
#include <cmath>

namespace multi_critical {

    inline boost::timer::cpu_timer overall_timer, sorting_timer,multi_chunk_timer,transformation_to_1_critical_timer,prepare_input_complex_timer,prepare_output_complex_timer, compute_f0_timer,compute_f1_timer,compute_h0_timer,sort_by_grades_timer,convert_to_graded_matrices_timer,output_file_timer,read_input_timer,mpfree_timer,
	compute_f2_timer,compute_h1_timer,compute_hh_timer,construct_boundaries_timer,
	test_timer_1,test_timer_2,test_timer_3,test_timer_4;
  
inline void initialize_timers() {
  overall_timer.start();
  overall_timer.stop();
  sorting_timer.start();
  sorting_timer.stop();
  multi_chunk_timer.start();
  multi_chunk_timer.stop();
  mpfree_timer.start();
  mpfree_timer.stop();
  transformation_to_1_critical_timer.start();
  transformation_to_1_critical_timer.stop();
  prepare_input_complex_timer.start();
  prepare_input_complex_timer.stop();
  prepare_output_complex_timer.start();
  prepare_output_complex_timer.stop();
  compute_f0_timer.start();
  compute_f0_timer.stop();
  compute_f1_timer.start();
  compute_f1_timer.stop();
  compute_h0_timer.start();
  compute_h0_timer.stop();
  sort_by_grades_timer.start();
  sort_by_grades_timer.stop();
  convert_to_graded_matrices_timer.start();
  convert_to_graded_matrices_timer.stop();
  output_file_timer.start();
  output_file_timer.stop();
  read_input_timer.start();
  read_input_timer.stop();
  compute_f2_timer.start();
  compute_f2_timer.stop();
  compute_h1_timer.start();
  compute_h1_timer.stop();
  compute_hh_timer.start();
  compute_hh_timer.stop();
  construct_boundaries_timer.start();
  construct_boundaries_timer.stop();
  test_timer_1.start();
  test_timer_1.stop();
  test_timer_2.start();
  test_timer_2.stop();
  test_timer_3.start();
  test_timer_3.stop();
  test_timer_4.start();
  test_timer_4.stop();
  
}

inline void pretty_print_timer(boost::timer::cpu_timer& timer,double total) {
  if(double(timer.elapsed().wall)/std::pow(10,9)<0.0001) {
    std::cout << "0" << std::endl;
  } else{
    std::cout << double(timer.elapsed().wall)/std::pow(10,9) << "     ( "  <<  double(timer.elapsed().wall)/total*100 << "% )" << std::endl;
  }
}

inline void print_timers(double total=double(overall_timer.elapsed().wall),bool print_overall=true) {
  if(print_overall) {
    std::cout << "Overall timer:                " << double(overall_timer.elapsed().wall)/std::pow(10,9) << std::endl;
  }
  std::cout << "Read input timer:             "; pretty_print_timer(read_input_timer,total);
  std::cout << "Transformation timer:         "; pretty_print_timer(transformation_to_1_critical_timer,total);
  std::cout << "Sorting timer:                "; pretty_print_timer(sorting_timer,total);
  std::cout << "Multi chunk timer:            "; pretty_print_timer(multi_chunk_timer,total);
  std::cout << "Mpfree timer:                 "; pretty_print_timer(mpfree_timer,total);
  std::cout << "Output file timer:            "; pretty_print_timer(output_file_timer,total);
  std::cout << " ----- Subroutines of transformation -------" << std::endl;
  double transform_time=transformation_to_1_critical_timer.elapsed().wall;
  std::cout << "Prepare input complex timer:  "; pretty_print_timer(prepare_input_complex_timer,transform_time);
  std::cout << "Prepare output complex timer: "; pretty_print_timer(prepare_output_complex_timer,transform_time);
  std::cout << "Compute f0 timer:             "; pretty_print_timer(compute_f0_timer,transform_time);
  std::cout << "Compute f1 timer:             "; pretty_print_timer(compute_f1_timer,transform_time);
  std::cout << "Compute h0 timer:             "; pretty_print_timer(compute_h0_timer,transform_time);
  std::cout << "Compute f2 timer:             "; pretty_print_timer(compute_f2_timer,transform_time);
  std::cout << "Compute h1 timer:             "; pretty_print_timer(compute_h1_timer,transform_time);
  std::cout << "Compute hh timer:             "; pretty_print_timer(compute_hh_timer,transform_time);
  std::cout << "Construct boundaries timer:   "; pretty_print_timer(construct_boundaries_timer,transform_time);
  std::cout << "Sort by grades timer:         "; pretty_print_timer(sort_by_grades_timer,transform_time);
  std::cout << "Convert to gr matrices timer: "; pretty_print_timer(convert_to_graded_matrices_timer,transform_time);
  std::cout << " ----- Test timer -------" << std::endl;
  std::cout << "Test timer 1:           "; pretty_print_timer(test_timer_1,total);
  std::cout << "Test timer 2:           "; pretty_print_timer(test_timer_2,total);
  std::cout << "Test timer 3:           "; pretty_print_timer(test_timer_3,total);
  std::cout << "Test timer 4:           "; pretty_print_timer(test_timer_4,total);
}

}
  
// END copied from multicritical/include/multi_critical/boost_timers.h
#endif

// BEGIN copied from multicritical/include/multi_critical/free_resolution.h
/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/


#include<algorithm>
#include <unordered_map>
#include <unordered_set>

// BEGIN copied from multicritical/include/multi_critical/basic.h
/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/


namespace multi_critical {

    using index = std::int64_t;

    inline bool verbose = false;
    inline bool very_verbose=false;

} // of namespace multi-critical
// END copied from multicritical/include/multi_critical/basic.h
// BEGIN copied from multicritical/include/multi_critical/Grade.h
/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/



namespace multi_critical {

    using Degree = graded_linalg::CoordinateDegree<double, 2>;
    using Upset = graded_linalg::r2upset<double>;
    using Degree_traits = graded_linalg::Degree_traits<Degree>;
    using Upset_traits = graded_linalg::Degree_traits<Upset>;

    inline index get_copy_smaller_equal_than(const Upset& grades, const Degree& g) {
	//std::cout << "Grade: " << g[0] << " " << g[1] << std::endl;
#if 0
	for(int i=0;i<grades.size();i++) {
#else
        for(int i=grades.size()-1;i>=0;i--) {
#endif
	    //std::cout << "Candidate: " << grades[i][0] << " " << grades[i][1] << std::endl;
	    if(Degree_traits::smaller_equal(grades[i],g)) {
		//std::cout << "Works" << std::endl;
		return i;
	    }
	}
	std::cerr.precision(std::numeric_limits<double>::max_digits10 - 1);
	std::cerr << "Warning, no smaller grade found for " << g[0] << " " << g[1] << "!" << std::endl;
	std::cerr << "Candidates were: ";
	for(auto& gr: grades) {
	    std::cerr << gr[0] << " " << gr[1] << ", ";
	}
	std::cerr << std::endl;
	return -1;
    }
		
    


    
} // of namespace multi_critical
// END copied from multicritical/include/multi_critical/Grade.h


namespace multi_critical {



    // Represents a linear map
    typedef std::unordered_map<index,std::vector<index>> Map;

    // The first index is the dimension
    typedef std::vector<Map> Dim_Map;

    // Computes g o f (x)
#if 0
    inline void compose_maps(Map& g, Map& f, index x, std::vector<index>& result) {
	//std::cout << "Compose starts"  << std::endl;
	std::vector<index>& fx = f[x];
	std::unordered_set<index> container;
	for(index y : fx) {
	    std::vector<index>& gfx = g[y];
	    for(index z : gfx) {
		if(!container.count(z)) {
		    container.insert(z);
		} else {
		    container.erase(z);
		}
	    }
	    //std::cout << "Now container has size " << container.size() << std::endl;
	}
	std::copy(container.begin(),container.end(),std::back_inserter(result));
	std::sort(result.begin(),result.end());
    }

#else
   inline void compose_maps(Map& g, Map& f, index x, std::vector<index>& result) {
	//std::cout << "Compose starts"  << std::endl;
	std::vector<index>& fx = f[x];
	std::vector<index> container;
	for(index y : fx) {
	    std::vector<index>& gfx = g[y];
	    std::copy(gfx.begin(),gfx.end(),std::back_inserter(container));
	}
	std::sort(container.begin(),container.end());
	// Remove duplicates
	index container_it=0;
	while(container_it<container.size()) {
	    if(container_it<container.size()-1 && container[container_it]==container[container_it+1]) {
		container_it+=2;
		continue;
	    }
	    result.push_back(container[container_it]);
	    container_it++;
	}

    } 
#endif
    
    inline void sum_of_composed_maps(Map& g1, Map& f1, Map& g2, Map& f2, index x, std::vector<index>& result) {
	std::vector<index> part1,part2;
	compose_maps(g1,f1,x,part1);
	compose_maps(g2,f2,x,part2);
	index it1=0,it2=0;
	index n1=part1.size();
	index n2=part2.size();
	while(it1<n1 || it2<n2) {
	    if(it1==n1) {
		result.push_back(part2[it2]);
		it2++;
		continue;
	    }
	    if(it2==n2) {
		result.push_back(part1[it1]);
		it1++;
		continue;
	    }
	    if(part1[it1]==part2[it2]) {
		it1++;
		it2++;
		continue;
	    }
	    if(part1[it1]<part2[it2]) {
		result.push_back(part1[it1]);
		it1++;
		continue;
	    }
	    if(part1[it1]>part2[it2]) {
		result.push_back(part2[it2]);
		it2++;
		continue;
	    }
				       
	}
    }
	    
	    
    // This object is needed to sort relations in the syzygy creation routine
    template<typename InputComplex, typename OutputComplex>
    struct Relation_sorter {

	InputComplex& input_complex;
	OutputComplex& output_complex;
	Dim_Map& p1;
	
	index dim;
	
	Relation_sorter(InputComplex& input_complex,
			OutputComplex& output_complex,
			Dim_Map& p1)
	    : input_complex(input_complex),output_complex(output_complex), p1(p1) {
	}

	// Needed to set to pick the right dimension of relations in the comparison
	void set_dim(index dim) {
	    this->dim=dim;
	}

	/* We don't need that part of the code because the relations are sorted
	   by index in their width
	
	index priority(index rel_idx) {
	    std::vector<index>& p1_image = p1[dim][rel_idx];
	    //std::cout << "Size of " << rel_idx << " is : " << p1_image.size() << std::endl;
	    assert(p1_image.size()==2);
	    assert(output_complex[dim-1][p1_image[0]].copy_of == output_complex[dim-1][p1_image[1]].copy_of);
	    index first_copy_idx = output_complex[dim-1][p1_image[0]].copy_idx;
	    index second_copy_idx = output_complex[dim-1][p1_image[1]].copy_idx;
	    assert(second_copy_idx > first_copy_idx);
	    return second_copy_idx - first_copy_idx;
	}
	
	bool operator() (index rel_idx1, index rel_idx2) {
	    index prio1 = priority(rel_idx1);
	    index prio2 = priority(rel_idx2);
	    if(prio1<prio2) {
		return true;
	    }
	    if(prio1>prio2) {
		return false;
	    }
	    return rel_idx1<rel_idx2;
	}
	*/

	void find_preimage_of_syzygies(std::vector<index>& image_of_map,
				       std::vector<index>& result) {
	    if(image_of_map.size()==0) {
		//std::cout << "Composition is empty" << std::endl;
		return;
	    }
	    //std::cout << "Comp is not empty, size= " << image_of_map.size() << ", do something" << std::endl;
	    
	    int image_it = 0;
	    index idx_of_curr_input_simp = output_complex[dim][image_of_map[image_it]].copy_of;
	    std::vector<index> rels_of_input_simp;
	    while(image_it<image_of_map.size()) {
		//std::cout << "Here, Size of " << image_of_map[image_it] << " is " << p1[dim][image_of_map[image_it]].size() << std::endl;
		assert(p1[dim][image_of_map[image_it]].size()==2);
		if(output_complex[dim][image_of_map[image_it]].copy_of==idx_of_curr_input_simp) {
		    rels_of_input_simp.push_back(image_of_map[image_it]);
		} else {
		    /*
		    {
			std::cout << "Collected " << rels_of_input_simp.size() << " relations to fill in " << std::endl;
			std::cout << "Instance: ";
			for(auto x : rels_of_input_simp) {
			    std::cout << x << " ";
			}
			std::cout << std::endl;
		    }
		    */
		    input_complex[dim-1][idx_of_curr_input_simp].get_syzygies_of_cycle(rels_of_input_simp,result);
		    //std::cout << "Syzygies found" << std::endl;
		    rels_of_input_simp.clear();
		    rels_of_input_simp.push_back(image_of_map[image_it]);
		    idx_of_curr_input_simp=output_complex[dim][image_of_map[image_it]].copy_of;
		}
		image_it++;
		
	    }
	    assert(!rels_of_input_simp.empty());
	    //std::cout << "Collected, at the end now, remaining " <<rels_of_input_simp.size() << " relations"   << std::endl;
	    input_complex[dim-1][idx_of_curr_input_simp].get_syzygies_of_cycle(rels_of_input_simp,result);
	    std::sort(result.begin(),result.end());
	    //std::cout << "Image consists of " << result.size() << " triangles" << std::endl;
	}
    };

    
    struct Input_simplex {
	Upset grades;
	std::vector<index> boundary;
	std::vector<index> copies;
	// relations[i] is the list of pairs (j,idx) with i<j and idx a relation joining i and j
	std::vector<std::vector<std::pair<index,index>>> relations;
	// Syzygy with relation idx as upper boundary is saved with key idx
	std::unordered_map<index,index> syzygies;
	// The two other relations bounding the syzygy are saved here
	std::unordered_map<index,std::pair<index,index>> lower_hull_of_syzygy;
	
	Input_simplex() {}
	template <typename Boundary>
	Input_simplex(const Upset& grades,
		      const Boundary& boundary)
            : grades(grades), boundary(boundary.begin(), boundary.end()) {}

	void get_path_of_relations(int i,int j, std::vector<index>& rels) {
	    assert(i<=j);
	    if(i==j) {
		return;
	    }
	    int next_copy=0;
	    int next_rel=0;
	    for(auto pair : relations[i]) {
		if(pair.first>j) {
		    break;
		}
		next_copy=pair.first;
		next_rel=pair.second;
	    }
	    rels.push_back(next_rel);
	    get_path_of_relations(next_copy,j,rels);
	}
	
	index get_relation_of_copies(int i, int j) {
	    assert(i>=0 && j>=0);
	    assert(i<copies.size());
	    assert(j<copies.size());
	    for(auto p : relations[i]) {
		if(p.first==j) {
		    return p.second;
		}
	    }
	    return -1;
	}

	void get_syzygies_of_cycle(std::vector<index>& rels, std::vector<index>& result) {
	    // Little optimization: If there are 3 edges, it must be a "simple" cycle,
	    // and we can avoid the priority queue
	    if(rels.size()==3) {
		assert(rels[0]<rels[1] && rels[1]<rels[2]);
		result.push_back(syzygies[rels[2]]);
		return;
	    }
	    
	    /*
	    std::cout << "Set up pq with " << rels.size() << " elements: ";
	    for(index i : rels) {
		std::cout << i << " ";
	    }
	    std::cout << std::endl;
	    */
	    std::priority_queue<index> queue(rels.begin(),rels.end());
	    //td::cout << "Done" << std::endl;
	    //int pushed_elements=0;
	    while(!queue.empty()) {
		index next=queue.top();
		queue.pop();
		if(queue.top()==next) {
		    //there was a duplicate, remove it and go on
		    queue.pop();
		    continue;
		}
		/*
		std::cout << "next=" << next << std::endl;
		  {
		
		  
		  std::cout << "Number of copies: " << copies.size() << std::endl;
		  
		  std::cout << "Relations: " << std::endl;
		  for(auto x : relations) {
		  for(auto y : x) {
		  std::cout << y.second << " ";
		  }
		  }
		  std::cout << std::endl;
		  
		  std::cout << "Syzygy Container contains: " << std::endl;
		  for(auto p: syzygies) {
		  std::cout << p.first << "-> " << p.second << std::endl;
		  }
		*/
	    

		result.push_back(syzygies[next]);
		//pushed_elements++;
		//std::cout << "Lower hull Container contains: " << std::endl;
		/*
		for(auto p : lower_hull_of_syzygy) {
		    std::cout << p.first << "-> " << p.second.first << " " << p.second.second << std::endl;
		}
		*/
		std::pair<index,index>& lower_hull = lower_hull_of_syzygy[syzygies[next]];
		//std::cout << "Pushing " << lower_hull.first << " and " << lower_hull.second << std::endl;
		queue.push(lower_hull.first);
		queue.push(lower_hull.second);
	    }
	    //std::cout << "Resulted in " << pushed_elements << " pushed triangles" << std::endl;
	}
	
    };

    inline void debug_print_input_complex(std::vector<std::vector<Input_simplex>>& complex) {
	for(int i=0;i<complex.size();i++) {
	    std::cout << "Dimension " << i << std::endl;
	    for(int j=0;j<complex[i].size();j++) {
		std::cout << "Dimension/Index: " << i << " " << j << std::endl; 
		Input_simplex& s = complex[i][j];
		std::cout << "Grades: " << std::endl;
		for(Degree& gr : s.grades) {
		    std::cout << gr[0] << " " << gr[1] << std::endl;
		}
		std::cout << "Boundary: " << std::flush;
		for(index idx : s.boundary) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
		std::cout << "Copies: " << std::flush;
		for(index idx : s.copies) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
	    }
	}
    }

    enum Type {
	GENERATOR,
	RELATION,
	SYZYGY
    };
    
    
    struct Output_simplex {
	Degree grade;
	std::vector<index> boundary;
	index copy_of;
	index copy_idx;
	// Can be a copy of a generator or a relation joining two copies
	Type type;
	Output_simplex() {}
	Output_simplex(Degree& g, std::vector<index> bd, index copy_idx) {
	    grade=g;
	    std::copy(bd.begin(),bd.end(),std::back_inserter(boundary));
	    copy_of=copy_idx;
	}
    };

    
    inline void debug_print_output_complex(std::vector<std::vector<Output_simplex>>& complex) {
	for(int i=0;i<complex.size();i++) {
	    std::cout << "Dimension " << i << std::endl;
	    for(int j=0;j<complex[i].size();j++) {
		std::cout << "Dimension/Index: " << i << " " << j << std::endl; 
		Output_simplex& s = complex[i][j];
		std::cout << "Grade: " << s.grade[0] << " " << s.grade[1] << std::endl;
		std::cout << "Boundary: " << std::flush;
		for(index idx : s.boundary) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
		std::cout << "Copy of: " << s.copy_of << std::endl;
		std::cout << "Type: ";
		if(s.type==GENERATOR) {
		    std::cout << "Generator" << std::endl;
		}
		if(s.type==RELATION) {
		    std::cout << "Relation" << std::endl;
		}
		if(s.type==SYZYGY) {
		    std::cout << "Syzygy" << std::endl;
		}
	    }
	}
    }

    struct Grade_sorter {

	std::vector<Output_simplex>& data;

	Grade_sorter(std::vector<Output_simplex>& data) : data(data) {}

	bool operator() (int i, int j) {
	    return Degree_traits::lex_order(data[i].grade,data[j].grade);
	}
    };


    
    inline void boundary_of_boundaries(std::vector<std::vector<Output_simplex>>& complex, int ell, int idx, std::vector<index>& bd_of_bd) {
	
	Output_simplex& output_simplex=complex[ell][idx];
	for(index bd_idx : output_simplex.boundary) {
	    Output_simplex& bd_output_simplex=complex[ell-1][bd_idx];
	    std::copy(bd_output_simplex.boundary.begin(),
		      bd_output_simplex.boundary.end(),
		      std::back_inserter(bd_of_bd));
	}
	std::sort(bd_of_bd.begin(),bd_of_bd.end());
    }
	
	
    inline void check_output_complex(std::vector<std::vector<Output_simplex>>& complex) {

	// Check that boundary of vertices is empty
	for(int idx=0;idx<complex[0].size();idx++) {
	    if(complex[0][idx].boundary.size()!=0) {
		std::cerr << "Boundary in dim 0 is not empty!!" << std::endl;
	    }
	}
	
	// Check whether \partial\partial=0
	for(int ell=2;ell<complex.size();ell++) {
	    for(int idx=0;idx<complex[ell].size();idx++) {
		Output_simplex& output_simplex=complex[ell][idx];
		std::vector<index> bd_of_bd;
		boundary_of_boundaries(complex,ell,idx,bd_of_bd);
		/*
		{
		    std::cout << "BD_OF_BD: ";
		    for(index idx : bd_of_bd) {
			std::cout << idx << " ";
		    }
		    std::cout << std::endl;
		}
		*/
		if(bd_of_bd.size()%2!=0) {
		    std::cerr << "Bd of bd has odd number!" << std::endl;
		}
		for(int j=0;j<bd_of_bd.size();j+=2) {
		    if(bd_of_bd[j]!=bd_of_bd[j+1]) {
			std::cerr << "Bd of bd is not zero for dim=" << ell << ", idx=" << idx << std::endl;
			std::cerr << "Type of simplex: " << output_simplex.type << std::endl;
			std::cerr << "Content: " ;
			for(index p : bd_of_bd) {
			    std::cout << p << " ";
			}
			std::cout << std::endl;
		    }
		}
	    }
	}

	// Check grade consistency
	for(int ell=1;ell<complex.size();ell++) {
	    for(int idx=0;idx<complex[ell].size();idx++) {
		Output_simplex& output_simplex=complex[ell][idx];
		Degree& gr = output_simplex.grade;
		for(index bd_idx : output_simplex.boundary) {
		    Output_simplex& bd_output_simplex=complex[ell-1][bd_idx];
		    Degree bd_gr = bd_output_simplex.grade;
		    if(!Degree_traits::smaller_equal(bd_gr,gr)) {
			std::cerr << "Simplex (" << ell << ", " << idx << ") has grade " << gr[0] << " " << gr[1] << " but has grade " << bd_gr[0] << " " << bd_gr[1] << " in its boundary" << std::endl;
		    }
		}
	    }
	}
	
    }


    	    
	
    
    
    /** Replace an upset-graded chain complex by a quasi-isomorphic free one.
     * Input degrees must be canonical upsets, and the input must be graded and
     * satisfy the chain equations. The input is unchanged. Both complexes use
     * grlina's d1, d2, ... ordering; the output's coordinates remain doubles.
     */
    template <typename Index, typename Storage>
    auto free_resolution(
        const graded_linalg::ChainComplex<graded_linalg::UpsetGradedSparseMatrix<double, Index, Storage>>& input,
        bool use_logpath = false) {
        using Matrix = graded_linalg::CoordGradedSparseMatrix<double, 2, Index>;
        using Complex = graded_linalg::ChainComplex<Matrix>;
        if (input.empty()) return Complex{};

        const int L = static_cast<int>(input.size() + 1);

	if(multi_critical::verbose) std::cout << "Number of levels: " << L << std::endl;

	std::vector<index> idx_of_level_output(L+2,0);

	// input_complex[i][j] = j-th simplex in level i
	typedef std::vector<std::vector<Input_simplex>> Input_complex;
	
	Input_complex input_complex;
	input_complex.resize(L);

	typedef std::vector<std::vector<Output_simplex>> Output_complex;
	
	Output_complex output_complex;
	output_complex.resize(L+2);

	// These are presentation maps. p1: Rels->Gens, p2: Syz->Rels
	// With the path approach, p2 is not needed
	Dim_Map p1, p2;
	p1.resize(L+1);
	p2.resize(L+2);
	
	// These are the boundary maps, f0: Gens->Gens, f1: Rels->Rels, f2: Syz->Syz
	Dim_Map f0, f1, f2;
	f0.resize(L);
	f1.resize(L+1);
	f2.resize(L+2);
	
	// The 1st level homotopy maps. h0:Gens->Rels, h1: Rels->Syz
	Dim_Map h0,h1;
	h0.resize(L);
	h1.resize(L+1);
	
	// The 2nd level homotopy map hh: Gens->Syz
	Dim_Map hh;
	hh.resize(L);


	

#if MULTI_CRITICAL_TIMERS
	 multi_critical::prepare_input_complex_timer.start();
#endif

	if(multi_critical::verbose) std::cout << "Prepare input_complex" << std::endl;
        for (int ell = 0; ell < L; ++ell) {
            if (multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
            const auto& matrix = input[ell == 0 ? 0 : ell - 1];
            const auto& degrees = ell == 0 ? matrix.row_degrees : matrix.col_degrees;
            input_complex[ell].reserve(degrees.size());
            for (std::size_t column = 0; column < degrees.size(); ++column) {
                if (ell == 0) {
                    input_complex[ell].emplace_back(degrees[column], std::vector<index>{});
                } else {
                    input_complex[ell].emplace_back(degrees[column], matrix.column(static_cast<Index>(column)));
                }
            }
        }

#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_input_complex_timer.stop();
#endif

	
	//debug_print_input_complex(input_complex);

	if(multi_critical::verbose) std::cout << "Build 1-critical Generators and presentation maps"
					      << std::endl;

#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_output_complex_timer.start();
#endif
	 
	for(int ell=0;ell<L;ell++) {
	    if(multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
	    for(index idx_of_input_simplex=0;idx_of_input_simplex<input_complex[ell].size();idx_of_input_simplex++) {

		Input_simplex& input_simplex =input_complex[ell][idx_of_input_simplex];
		Upset& grades = input_simplex.grades;
		//std::cout << "New input simplex with " << grades.size() << " copies" << std::endl;		
		//std::cout << "Generator is " << grades.size() << "-critical" << std::endl;
		// Generate the basis elements
		input_simplex.copies.reserve(grades.size());
		for(int j=0;j<grades.size();j++) {
		    Degree& gr=grades[j];
		    index idx_of_new_output_simplex=idx_of_level_output[ell];
		    Output_simplex new_output_simplex;
		    new_output_simplex.type=GENERATOR;
		    new_output_simplex.grade=gr;
		    new_output_simplex.copy_of=idx_of_input_simplex;
		    new_output_simplex.copy_idx=j;
		    // Boundary is determined later
		    assert(idx_of_new_output_simplex>=0);
		    input_simplex.copies.push_back(idx_of_new_output_simplex);
		    output_complex[ell].push_back(new_output_simplex);
		    idx_of_level_output[ell]++;
		}
		// Generate relations and syzygies
		int width=1;
		input_simplex.relations.resize(input_simplex.copies.size());
		// It is important for later that the relations are generated in increasing
		// length, that is, the width of the relations (for a single input simplex)
		// increase with their index
		while(width < grades.size()) {
		    for(int j=width;j<grades.size();j+=width) {
			//std::cout << "Creating relation " << j-width << " " << j << std::endl;
			Output_simplex new_output_simplex_for_relation;
			index idx_of_new_output_relation=idx_of_level_output[ell+1];
			idx_of_level_output[ell+1]++;
			new_output_simplex_for_relation.type=RELATION;
			new_output_simplex_for_relation.grade=Degree_traits::join(grades[j-width],grades[j]);
			new_output_simplex_for_relation.copy_of=idx_of_input_simplex;
			assert(input_simplex.copies[j-width] < input_simplex.copies[j]);
			//std::cout << "HERE " << input_simplex.copies[j-width] << " " << input_simplex.copies[j] << std::endl;
			std::vector<index> image;
			image.push_back(input_simplex.copies[j-width]);
			image.push_back(input_simplex.copies[j]);
			p1[ell+1][idx_of_new_output_relation]=std::move(image);
						
			output_complex[ell+1].push_back(new_output_simplex_for_relation);
			input_simplex.relations[j-width].push_back(std::make_pair(j,idx_of_new_output_relation));
			
			
			if(width>1) {
			    Output_simplex new_output_simplex_for_syzygy;
			    index idx_of_new_output_syzygy=idx_of_level_output[ell+2];
			    idx_of_level_output[ell+2]++;
			    new_output_simplex_for_syzygy.type=SYZYGY;
			    new_output_simplex_for_syzygy.grade=new_output_simplex_for_relation.grade;
			    new_output_simplex_for_syzygy.copy_of=idx_of_input_simplex;
			    index lower_rel_idx_1 = input_simplex.get_relation_of_copies(j-width,j-width/2);
			    index lower_rel_idx_2 = input_simplex.get_relation_of_copies(j-width/2,j);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(lower_rel_idx_1);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(lower_rel_idx_2);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(idx_of_new_output_relation);
			    assert(p2[ell+2][idx_of_new_output_syzygy][0]<p2[ell+2][idx_of_new_output_syzygy][1]);
			    assert(p2[ell+2][idx_of_new_output_syzygy][1]<p2[ell+2][idx_of_new_output_syzygy][2]);
			    output_complex[ell+2].push_back(new_output_simplex_for_syzygy);
			    input_simplex.syzygies[idx_of_new_output_relation]=idx_of_new_output_syzygy;
			    input_simplex.lower_hull_of_syzygy[idx_of_new_output_syzygy]=std::make_pair(lower_rel_idx_1,lower_rel_idx_2);
			}
		    }
		    if(!use_logpath) {
			// Only use path, so width>1 is not allowed
			break;
		    }
		    width*=2;
		}
	    }
	}
	

#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_output_complex_timer.stop();
#endif
	

	
	//debug_print_input_complex(input_complex);
	    
	
	// Determine the rest of the boundaries

	// First choose a consistent map between the generators
	if(multi_critical::verbose) std::cout << "Constructing map f0" << std::endl;

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f0_timer.start();
#endif
	
	
	for(int ell=1;ell<L;ell++) {
	    if(multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		std::vector<index>& bd = input_simplex.boundary;
		/*
		{
		    std::cout << "Here I am at " << ell << ", " << std::endl;
		    
		    
		    std::cout << "Boundary (" << bd.size() << "): ";
		    for(index x : bd ) {
			std::cout << x << " " << std::flush;
		    }
		    std::cout << std::endl;
		}
		*/
		for(int i=0; i< input_simplex.copies.size(); i++) {
		    index output_simplex_idx = input_simplex.copies[i];
		    Output_simplex& output_simplex = output_complex[ell][output_simplex_idx];
		    assert(output_simplex.type==GENERATOR);
		    Degree& gr_of_copy = output_simplex.grade;
		    std::vector<index> image;
		    for(index idx : bd) {
			//std::cout << "bd index: " << idx << std::endl;
			Input_simplex& bd_input_simplex=input_complex[ell-1][idx];
			assert(bd_input_simplex.grades.size()>=1);

			index idx_copy =get_copy_smaller_equal_than(bd_input_simplex.grades,gr_of_copy);
			assert(bd_input_simplex.copies[idx_copy]>=0);
			image.push_back(bd_input_simplex.copies[idx_copy]);
			//std::cout << "Adding to bd on level " << ell << std::endl;
		    }
		    f0[ell][output_simplex_idx]=std::move(image);
		}
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f0_timer.stop();
#endif


	
	//debug_print_output_complex(output_complex);
	
	
	// Now compute the map between relations
	if(multi_critical::verbose) std::cout << "Constructing map f1" << std::endl;
	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f1_timer.start();
#endif

	for(int ell=L-1;ell>=1;ell--) {
	    long count_inst=0;		
	    if(multi_critical::very_verbose) std::cout << "Dim ell=" << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		std::vector<index>& bd = input_simplex.boundary;
		for(int i=0; i<input_simplex.copies.size(); i++) {
		    for(auto p : input_simplex.relations[i]) {
			count_inst++;
			int j = p.first;
			assert(i<j);
			index relation_idx = p.second;
			Output_simplex& output_relation = output_complex[ell+1][relation_idx];
		    
			assert(output_relation.type==RELATION);
			// Get the two generators that create the relation

			index output_gen_smaller_idx = input_simplex.copies[i];
			index output_gen_larger_idx = input_simplex.copies[j];
			Output_simplex& output_gen_smaller = output_complex[ell][output_gen_smaller_idx];
			Output_simplex& output_gen_larger = output_complex[ell][output_gen_larger_idx];

			std::vector<index>& p1_of_rel = p1[ell+1][relation_idx];
			assert(p1_of_rel.size()==2);
			assert(p1_of_rel[0]==output_gen_smaller_idx);
			assert(p1_of_rel[1]==output_gen_larger_idx);
			    
			assert(output_gen_smaller.type==GENERATOR);
			assert(output_gen_larger.type==GENERATOR);
			std::vector<index> combined_bd;
			std::copy(f0[ell][output_gen_smaller_idx].begin(),f0[ell][output_gen_smaller_idx].end(),std::back_inserter(combined_bd));
			std::copy(f0[ell][output_gen_larger_idx].begin(),f0[ell][output_gen_larger_idx].end(),std::back_inserter(combined_bd));
			std::sort(combined_bd.begin(),combined_bd.end());

			/*
			{
			    std::cout << "Info (" << ell << "): " << std::flush;
			    
			    for(index idx : combined_bd) {
				std::cout << idx << " ";
			    }
			    std::cout << std::endl;
			}
			*/

			std::vector<index> image;
			for(int j=0;j<combined_bd.size();j+=2) {
			    Output_simplex& first_copy = output_complex[ell-1][combined_bd[j]];
			    Output_simplex& second_copy = output_complex[ell-1][combined_bd[j+1]];
			    //std::cout << "Copies " << combined_bd[j] << " " << combined_bd[j+1] << std::endl;
			    //std::cout << "Copy of " << first_copy.copy_of << " " << second_copy.copy_of << std::endl;
			    assert(first_copy.copy_of==second_copy.copy_of);
			    assert(first_copy.copy_idx<=second_copy.copy_idx);
			    assert(first_copy.type==GENERATOR);
			    assert(second_copy.type==GENERATOR);
			    Input_simplex& input_simplex = input_complex[ell-1][first_copy.copy_of];
			    assert(input_simplex.copies[first_copy.copy_idx]==combined_bd[j]);
			    assert(input_simplex.copies[second_copy.copy_idx]==combined_bd[j+1]);

			    input_simplex.get_path_of_relations(first_copy.copy_idx, second_copy.copy_idx, image);
			}
			std::sort(image.begin(),image.end());
			f1[ell+1][relation_idx]=std::move(image);
		    }
		}
	    }
	    if(multi_critical::very_verbose) std::cout << "Handled " << count_inst << " instances" << std::endl;
	}


#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f1_timer.stop();
#endif

	if(multi_critical::verbose) std::cout << "Constructing map h0 " << std::endl;
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h0_timer.start();
#endif
	// And finally the map from generators to relations in codim 2
	for(int ell=L-1;ell>=2;ell--) {
	    long count_inst=0;
	    if(multi_critical::very_verbose) std::cout << "At dimension " << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		/*
		{
		    std::cout << "AT: ";
		    for(index idx : input_simplex.boundary) {
			std::cout << idx << " ";
		    }
		    std::cout << std::endl;
		}
		*/
		for(int i=0; i<input_simplex.copies.size(); i++) {
		    count_inst++;
		    index copy_idx = input_simplex.copies[i];
		    std::vector<index> h_circ_h_of_copy_idx;
		    compose_maps(f0[ell-1],f0[ell],copy_idx,h_circ_h_of_copy_idx);
		    /*
		    std::cout << "Size of hh_copy_idx: " << h_circ_h_of_copy_idx.size() << std::endl;
		    
		    {
			std::cout << "Info2 (" << ell << "): " << std::flush;
			
			for(index idx : h_circ_h_of_copy_idx) {
			    std::cout << idx << " ";
			}
			std::cout << std::endl;
		    }
		    */
		    std::vector<index> image;
		    for(int j=0;j<h_circ_h_of_copy_idx.size();j+=2) {
			Output_simplex& first_copy = output_complex[ell-2][h_circ_h_of_copy_idx[j]];
			Output_simplex& second_copy = output_complex[ell-2][h_circ_h_of_copy_idx[j+1]];
			assert(first_copy.type==GENERATOR);
			assert(second_copy.type==GENERATOR);
			assert(first_copy.copy_of==second_copy.copy_of);
			assert(first_copy.copy_idx<=second_copy.copy_idx);
			Input_simplex& input_simplex_of_bd = input_complex[ell-2][first_copy.copy_of];
			assert(input_simplex_of_bd.copies[first_copy.copy_idx]==h_circ_h_of_copy_idx[j]);
			assert(input_simplex_of_bd.copies[second_copy.copy_idx]==h_circ_h_of_copy_idx[j+1]);
			input_simplex_of_bd.get_path_of_relations(first_copy.copy_idx, second_copy.copy_idx, image);
			
		    }
		    std::sort(image.begin(),image.end());
		    h0[ell][copy_idx]=std::move(image);
	    	}
	    }
	    if(multi_critical::very_verbose) std::cout << "Handled " << count_inst << " instances" << std::endl;
	}
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h0_timer.stop();
#endif

	Relation_sorter<Input_complex, Output_complex> relation_sorter(input_complex,output_complex,p1);


	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f2_timer.start();
#endif
	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute f2" << std::endl;
	    
	    for(int ell=L+1;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		long no_elements_with_non_trivial_preimage=0;
		
		relation_sorter.set_dim(ell-2);
		
		// Iterating over all syzygies is very inconvenient and indirect.
		// TODO: Store the output complex separately in the fields
		// generators, relations, syzygies, each indexed by dimension,
		// and only merge in the end
		for(Input_simplex& input_simplex : input_complex[ell-2]) {
		    for(auto& p : input_simplex.syzygies) {
			index idx_of_syz = p.second;
			std::vector<index> p_circ_f;
			compose_maps(f1[ell-1],p2[ell],idx_of_syz,p_circ_f);
			std::vector<index> preimage;
			relation_sorter.find_preimage_of_syzygies(p_circ_f,preimage);
			if(!preimage.empty()) {
			    f2[ell][idx_of_syz]=preimage;
			    no_elements_with_non_trivial_preimage++;
			}
		    }
		    
		}
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f2_timer.stop();
#endif

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h1_timer.start();
#endif

	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute h1" << std::endl;
	    
	    for(int ell=L;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		relation_sorter.set_dim(ell-2);
		
		long no_elements_with_non_trivial_preimage=0;
		std::vector<index> preimage;
		
		for(Input_simplex& input_simplex : input_complex[ell-1]) {
		    for(int i=0; i<input_simplex.copies.size(); i++) {
			for(auto& p : input_simplex.relations[i]) {
			    int j = p.first;
			    assert(i<j);
			    index relation_idx = p.second;
			    //std::cout << "Handling relation_idx " << relation_idx << std::endl;
			    std::vector<index> image_of_map;
#if MULTI_CRITICAL_TIMERS
			    test_timer_1.resume();
#endif
			    sum_of_composed_maps(f1[ell-1],f1[ell],h0[ell-1],p1[ell],relation_idx,image_of_map);
#if MULTI_CRITICAL_TIMERS
			    test_timer_1.stop();
#endif
			    //compose_maps(f1[ell-1],f1[ell],relation_idx,image_of_map);
			    //compose_maps(h0[ell-1],p1[ell],relation_idx,image_of_map);
			    // Now image_of_map is the sum of the two composed maps.
			    //std::cout << "Size of image is " << image_of_map.size() << std::endl;
			    preimage.clear();
#if MULTI_CRITICAL_TIMERS
			    test_timer_2.resume();
#endif
			    relation_sorter.find_preimage_of_syzygies(image_of_map,preimage);
#if MULTI_CRITICAL_TIMERS
			    test_timer_2.stop();
#endif
			    if(!preimage.empty()) {
				h1[ell][relation_idx]=preimage;
				no_elements_with_non_trivial_preimage++;
			    }
			}
		    }
	
		}
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}

	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h1_timer.stop();
#endif


#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_hh_timer.start();
#endif

	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute hh" << std::endl;
	    
	    for(int ell=L-1;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		relation_sorter.set_dim(ell-2);
		
		long no_elements_with_non_trivial_preimage=0;
		
		for(Input_simplex& input_simplex : input_complex[ell]) {
		    /*
		      {
		      std::cout << "AT: ";
		      for(index idx : input_simplex.boundary) {
		      std::cout << idx << " ";
		      }
		      std::cout << std::endl;
		      }
		    */
		    for(int i=0; i<input_simplex.copies.size(); i++) {
			index gen_idx = input_simplex.copies[i];
			
			std::vector<index> image_of_map;
			sum_of_composed_maps(h0[ell-1],f0[ell],f1[ell-1],h0[ell],gen_idx,image_of_map);
			
			std::vector<index> preimage;
			relation_sorter.find_preimage_of_syzygies(image_of_map,preimage);
			
			if(!preimage.empty()) {
			    hh[ell][gen_idx]=preimage;
			    no_elements_with_non_trivial_preimage++;
			}
		    }
		}
		
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_hh_timer.stop();
#endif

#if MULTI_CRITICAL_TIMERS
	multi_critical::construct_boundaries_timer.start();
#endif

	if(multi_critical::verbose) std::cout << "Construct the boundaries" << std::endl;
	
	// Now construct the boundaries of the chain complex

	for(int ell=L+1;ell>=0;ell--) {
	    for(index idx=0;idx<output_complex[ell].size();idx++) {
		Output_simplex& output_simplex = output_complex[ell][idx];
		assert(output_simplex.boundary.empty());
		//std::cout << "ell=" << ell << ", idx=" << idx << ", type:  " << output_simplex.type << " ";
		if(output_simplex.type==GENERATOR) {
		    output_simplex.boundary.reserve(f0[ell][idx].size() + h0[ell][idx].size() + hh[ell][idx].size());
		    std::copy(f0[ell][idx].begin(),f0[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(h0[ell][idx].begin(),h0[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(hh[ell][idx].begin(),hh[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		}
		if(output_simplex.type==RELATION) {
		    output_simplex.boundary.reserve(p1[ell][idx].size() + f1[ell][idx].size() + h1[ell][idx].size());
		    std::copy(p1[ell][idx].begin(),p1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(f1[ell][idx].begin(),f1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(h1[ell][idx].begin(),h1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		}
		if(output_simplex.type==SYZYGY) {
		    output_simplex.boundary.reserve(p2[ell][idx].size() + f2[ell][idx].size());
		    std::copy(p2[ell][idx].begin(),p2[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(f2[ell][idx].begin(),f2[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    // Map to gen is 0
		}
		/*
		std::cout << "Bd: ";
		for(index p : output_simplex.boundary) {
		    std::cout << p << " ";
		}
		std::cout << std::endl;
		*/
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::construct_boundaries_timer.stop();
#endif

	
	// Output complex is done - check whether it is a graded cell complex

#if !NDEBUG
	if(multi_critical::verbose) std::cout << "Checking.." << std::endl;
	check_output_complex(output_complex);

	if(multi_critical::verbose) std::cout << "done checking" << std::endl;
#endif

	// ...and convert to graded matrices

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.start();
#endif

	std::vector<std::vector<index>> permutations, inv_permutations;
	permutations.resize(output_complex.size());
	inv_permutations.resize(output_complex.size());
	for(int i=0;i<output_complex.size();i++) {
	    long m = output_complex[i].size();
	    permutations[i].resize(m);
	    inv_permutations[i].resize(m);
	    for(int j=0;j<m;j++) {
		permutations[i][j]=j;
	    }
#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::stable_sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
	    for(int j=0;j<m;j++) {
		inv_permutations[i][permutations[i][j]]=j;
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.stop();
#endif
	
	
#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.start();
#endif
	if(multi_critical::verbose) std::cout << "Convert to graded matrices" << std::endl;
        // Each output group is already stably sorted by its degree. Build
        // d1, d2, ... directly, sharing that ordering between adjacent matrices.
        // C0 is represented by d1's rows, without a separate C0 -> 0 matrix.
        for (const auto& group : output_complex) {
            if (group.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
                throw std::overflow_error("Free resolution exceeds the matrix index range");
        }
        std::vector<Matrix> matrices;
        matrices.reserve(output_complex.size() - 1);
        for (std::size_t ell = 1; ell < output_complex.size(); ++ell) {
            const auto columns = static_cast<Index>(output_complex[ell].size());
            const auto rows = static_cast<Index>(output_complex[ell - 1].size());
            Matrix matrix(columns, rows);
            for (Index row = 0; row < rows; ++row)
                matrix.row_degrees[row] = output_complex[ell - 1][permutations[ell - 1][row]].grade;
            for (Index column = 0; column < columns; ++column) {
                const auto& simplex = output_complex[ell][permutations[ell][column]];
                matrix.col_degrees[column] = simplex.grade;
                std::vector<Index> boundary;
                boundary.reserve(simplex.boundary.size());
                for (index row : simplex.boundary)
                    boundary.push_back(static_cast<Index>(inv_permutations[ell - 1][row]));
                std::sort(boundary.begin(), boundary.end());
                matrix.set_col(column, std::move(boundary));
            }
            matrix.compatibly_sorted = true;
            matrices.push_back(std::move(matrix));
        }

#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.stop();
#endif


	
	return Complex(std::move(matrices));
    }
    
    
}
// END copied from multicritical/include/multi_critical/free_resolution.h

