#pragma once

// Consolidated upstream headers; implementation and namespaces are unchanged.

// BEGIN copied from multicritical/phat_mod/include/phat/boundary_matrix.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */


// BEGIN copied from multicritical/phat_mod/include/phat/helpers/misc.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */


// STL includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include <queue>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <iterator>

// VS2008 and below unfortunately do not support stdint.h
#if defined(_MSC_VER)&& _MSC_VER < 1600
    typedef __int8 int8_t;
    typedef unsigned __int8 uint8_t;
    typedef __int16 int16_t;
    typedef unsigned __int16 uint16_t;
    typedef __int32 int32_t;
    typedef unsigned __int32 uint32_t;
    typedef __int64 int64_t;
    typedef unsigned __int64 uint64_t;
#else
    #include <stdint.h>
#endif

// basic types. index can be changed to int32_t to save memory on small instances
namespace phat {
    typedef int64_t index;
    typedef int8_t dimension;
    typedef std::vector< index > column;
}

// OpenMP (proxy) functions
#if defined _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
    #define omp_get_num_threads() 1
	inline void omp_set_num_threads( int ) {};
    #include <time.h>
    #define omp_get_wtime() (float)clock() / (float)CLOCKS_PER_SEC
#endif

// BEGIN copied from multicritical/phat_mod/include/phat/helpers/thread_local_storage.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



// should ideally be equal to the cache line size of the CPU
#define PHAT_TLS_SPACING_FACTOR 64

// ThreadLocalStorage with some spacing to avoid "false sharing" (see wikipedia)
template< typename T > 
class thread_local_storage
{
public:

    thread_local_storage() : per_thread_storage( omp_get_max_threads() * PHAT_TLS_SPACING_FACTOR ) {};

    T& operator()() {
        return per_thread_storage[ omp_get_thread_num() * PHAT_TLS_SPACING_FACTOR ];
    }

    const T& operator()() const {
        return per_thread_storage[ omp_get_thread_num() * PHAT_TLS_SPACING_FACTOR ];
    }

    T& operator[]( int tid ) {
        return per_thread_storage[ tid * PHAT_TLS_SPACING_FACTOR ];
    }

    const T& operator[]( int tid ) const {
        return per_thread_storage[ tid * PHAT_TLS_SPACING_FACTOR ];
    }

protected:
    std::vector< T > per_thread_storage; 
};
// END copied from multicritical/phat_mod/include/phat/helpers/thread_local_storage.h



// END copied from multicritical/phat_mod/include/phat/helpers/misc.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/default_representations.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */


// BEGIN copied from multicritical/phat_mod/include/phat/representations/Uniform_representation.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */


// BEGIN copied from multicritical/phat_mod/include/phat/representations/Container_traits.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */


namespace phat {
  
template<typename ColumnContainer>
  class Column_container_traits {
 
  public:  

  typedef ColumnContainer Column_container;
  typedef typename Column_container::value_type Column_type;

  Column_type& col_at(Column_container& data, index idx) const {
    return data[ idx ];
  }

  const Column_type& col_at(const Column_container& data, index idx) const {
    return data[ idx ];
  }

  index get_size(const Column_container& data) const {
    return data.size();
  }

  void resize(Column_container& data, index nr_of_columns) const {
    data.resize(nr_of_columns);
  }

};

template<typename DimensionContainer>
  class Dimension_container_traits {
  
  public:

  typedef DimensionContainer Dimension_container;

  index& dim_at(Dimension_container& data, index idx) const {
    return data[ idx ];
  }

  const index& dim_at(const Dimension_container& data, index idx) const {
    return data[ idx ];
  }

  index get_size(const Dimension_container& data) const {
    return data.size();
  }

  void resize(Dimension_container& data, index nr_of_columns) const { 
    data.resize(nr_of_columns);
  }

};


} // namespace phat
// END copied from multicritical/phat_mod/include/phat/representations/Container_traits.h

namespace phat {
  
template<class ColumnContainer, class DimensionContainer>
class Uniform_representation {
  
 public:

  typedef ColumnContainer Column_container;
  typedef DimensionContainer Dimension_container;

  typedef Column_container_traits<Column_container> Column_traits;
  typedef Dimension_container_traits<Dimension_container> Dimension_traits;


  typedef typename Column_traits::Column_type Column_type;

 protected:

  // We need to label them mutable, since we have no control whether
  // the representations are manipulated by the traits class
  mutable Dimension_container dims;
  mutable Column_container matrix;

  Column_traits col_traits;
  Dimension_traits dim_traits;

  thread_local_storage< column > temp_column_buffer;

 public:
  
  index _get_num_cols() const {
    return col_traits.get_size(matrix);
  }
  
  void _set_dimensions(index nr_of_rows, index nr_of_columns) {
    col_traits.resize(matrix, nr_of_columns);
    for(index idx = 0;idx < nr_of_columns;idx++) {
      col_traits.col_at(matrix,idx)._set_nr_of_rows(nr_of_rows);
      col_traits.col_at(matrix,idx).offer_thread_local_storage(&temp_column_buffer);
    }
    dim_traits.resize(dims, nr_of_columns);
  }

  dimension _get_dim( index idx ) const {
    return dim_traits.dim_at(dims, idx );
  }

  void _set_dim( index idx, dimension dim ) { 
    dim_traits.dim_at(dims,idx) = dim; 
  }

  void _get_col( index idx, column& col  ) const { 
    col_traits.col_at(matrix, idx)._get_col( col ); 
  }

  void _set_col( index idx, const column& col  ) { 
    //col_traits.col_at(matrix, idx)._set_col( col );
    matrix[idx]._set_col(col);
  }
  
  bool _is_empty( index idx ) const {
    return col_traits.col_at(matrix, idx)._is_empty();
  }

  index _get_max_index( index idx ) const { 
    return col_traits.col_at(matrix, idx)._get_max_index(); 
  }

  void _remove_max( index idx ) { 
    return col_traits.col_at(matrix, idx)._remove_max(); 
  }

  void _add_to( index source, index target ) { 
    Column_type& source_col = col_traits.col_at(matrix, source);
    Column_type& target_col = col_traits.col_at(matrix, target);
    target_col._add_to( source_col ); 
  }

  void _clear( index idx ) { 
    col_traits.col_at(matrix, idx)._clear(); 
  }
        
  void _finalize( index idx ) { 
    col_traits.col_at(matrix, idx)._finalize(); 
  }
        
  void _sync() {}

};
 
}
// END copied from multicritical/phat_mod/include/phat/representations/Uniform_representation.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/Pivot_representation.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat{

template<typename BaseRepresentation, typename PivotColumn> 
  class Pivot_representation : public BaseRepresentation {

 public:
  
  typedef BaseRepresentation Base;
  typedef PivotColumn Pivot_column;
  
 protected:
  
  // For parallization purposes, it could be more than one full column
  mutable thread_local_storage< Pivot_column > pivot_cols;
  mutable thread_local_storage< index > idx_of_pivot_cols;

  Pivot_column& get_pivot_col() const {
    return pivot_cols();
  }

  bool is_pivot_col( index idx ) const {
    return idx_of_pivot_cols() == idx;
  }
      
  void release_pivot_col() {
    index idx = idx_of_pivot_cols();
    if( idx != -1 ) {
      Base::_clear(idx);
      column col;
      pivot_cols().get_col_and_clear( col );
      Base::_set_col(idx, col);
    }
    idx_of_pivot_cols() = -1;
  }
  
  void make_pivot_col( index idx ) {
    release_pivot_col();
    idx_of_pivot_cols() = idx;
    // Remark: Converting the column to a vector first gives a huge performance
    //         penalty, due to the fact that data is copied around
    //column col;
    //Base::_get_col( idx, col );
    
    get_pivot_col().add_col( Base::col_traits.col_at(Base::matrix,idx).begin(),
			     Base::col_traits.col_at(Base::matrix,idx).end());
    
    //get_pivot_col().add_col( Base::matrix[idx]);
  }
  
 public:  

  void _set_dimensions( index nr_of_rows, index nr_of_cols ) {
    #pragma omp parallel for
    for( int tid = 0; tid < omp_get_num_threads(); tid++ ) {
      pivot_cols[ tid ].init( nr_of_rows );
      idx_of_pivot_cols[ tid ] = -1;
    }
    Base::_set_dimensions( nr_of_rows, nr_of_cols );
  }

  void _add_to( index source, index target ) {
    if( !is_pivot_col( target ) )
      make_pivot_col( target );
    //column col;
    //Base::_get_col(source,col);
    
    get_pivot_col().add_col( Base::col_traits.col_at(Base::matrix,source).begin(),
			     Base::col_traits.col_at(Base::matrix,source).end() );
    
    //get_pivot_col().add_col(Base::matrix[source]);
  }

  void _sync() { 
    #pragma omp parallel for
    for( int tid = 0; tid < omp_get_num_threads(); tid++ )
      release_pivot_col();
  } 

  void _get_col( index idx, column& col  ) const { is_pivot_col( idx ) ? get_pivot_col().get_col( col ) : Base::_get_col( idx, col ); }
        
  bool _is_empty( index idx ) const { return is_pivot_col( idx ) ? get_pivot_col().is_empty() : Base::_is_empty( idx ); }

  index _get_max_index( index idx ) const { return is_pivot_col( idx ) ? get_pivot_col().get_max_index() : Base::_get_max_index( idx ); }

  void _clear( index idx ) { is_pivot_col( idx ) ? get_pivot_col().clear() : Base::_clear( idx ); }

  void _set_col( index idx, const column& col  ) { is_pivot_col( idx ) ? get_pivot_col().set_col( col ) : Base::_set_col( idx, col ); }

  void _remove_max( index idx ) { is_pivot_col( idx ) ? get_pivot_col().remove_max() : Base::_remove_max( idx ); }
        
  void finalize( index idx ) { Base::_finalize( idx ); }

};


}
// END copied from multicritical/phat_mod/include/phat/representations/Pivot_representation.h

// BEGIN copied from multicritical/phat_mod/include/phat/representations/sparse_pivot_column.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class sparse_column {

    protected:
        std::set< index > data;

        void add_index( const index idx ) {
            std::pair< std::set< index >::iterator, bool > result = data.insert( idx );
            if( result.second == false )
                data.erase( result.first );
        }

    public:
        void init( const index total_size ) {
            data.clear(); 
        }
	
	
	template<typename InputIterator>
	  void add_col(InputIterator begin, InputIterator end) {
	  for(InputIterator it=begin;it!=end;it++) {
	    add_index(*it);
	  }
	}
	

        void add_col( const column& col ) {
            for( index idx = 0; idx < (index) col.size(); idx++ )
                add_index( col[ idx ] );
        }

        index get_max_index() {
            return data.empty() ? -1 : *data.rbegin();
        }

        void get_col_and_clear( column& col ) {
            col.assign( data.begin(), data.end() );
            data.clear();
        }

        bool is_empty() {
            return data.empty();
        }

		void clear() {
			data.clear();
		}

		void remove_max() {
            add_index( get_max_index() );
        }

        void set_col( const column& col  ) {
            clear();
            add_col( col );
        }

        void get_col( column& col  ) {
            get_col_and_clear( col );
            add_col( col );
        }
    };

}
// END copied from multicritical/phat_mod/include/phat/representations/sparse_pivot_column.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/full_pivot_column.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class full_column {

    protected:
        std::priority_queue< index > history;
        std::vector< char > is_in_history;
        std::vector< char > col_bit_field;

    public:
        void init( const index total_size ) {
            col_bit_field.resize( total_size, false );
            is_in_history.resize( total_size, false );
        }

	
	template<typename InputIterator>
	  void add_col(InputIterator begin, InputIterator end) {
	  for(InputIterator it=begin;it!=end;it++) {
	    add_index(*it);
	  }
	}
	

        void add_col( const column& col ) {
            for( index idx = 0; idx < (index) col.size(); idx++ ) {
                add_index( col[ idx ] );
            }
        }

        void add_index( const index idx ) {
            if( !is_in_history[ idx ] ) {
                history.push( idx );
                is_in_history[ idx ] = true; 
            }

            col_bit_field[ idx ] = !col_bit_field[ idx ];
        }

        index get_max_index() {
            while( history.size() > 0 ) {
                index topIndex = history.top();
                if( col_bit_field[ topIndex ] ) {
                    return topIndex;
                } else {
                    history.pop();
                    is_in_history[ topIndex ] = false;
                }
            }
    
            return -1;
        }

        void get_col_and_clear( column& col ) {
            while( !is_empty() ) {
                col.push_back( get_max_index() );
                add_index( get_max_index() );
            }
            std::reverse( col.begin(), col.end() );
        }

        bool is_empty() {
            return (get_max_index() == -1);   
        }

		void clear() {
			while( !is_empty() )
				add_index( get_max_index() );
		}

		void remove_max() {
            add_index( get_max_index() );
        }

        void set_col( const column& col  ) {
            clear();
            add_col( col );
        }

        void get_col( column& col  ) {
            get_col_and_clear( col );
            add_col( col );
        }
    };

}
// END copied from multicritical/phat_mod/include/phat/representations/full_pivot_column.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/heap_pivot_column.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class heap_column {

    protected:
        std::priority_queue< index > data;

        column temp_col;
        index inserts_since_last_prune;

        void prune()
        {
            temp_col.clear( );
            index max_index = pop_max_index( );
            while( max_index != -1 ) {
                temp_col.push_back( max_index );
                max_index = pop_max_index( );
            }

            for( index idx = 0; idx < (index)temp_col.size( ); idx++ )
                data.push( temp_col[ idx ] );

            inserts_since_last_prune = 0;
        }

        index pop_max_index()
        {
            if( data.empty( ) )
                return -1;
            else {
                index max_element = data.top( );
                data.pop();
                while( !data.empty( ) && data.top( ) == max_element ) {
                    data.pop( );
                    if( data.empty( ) )
                        return -1;
                    else {
                        max_element = data.top( );
                        data.pop( );
                    }
                }
                return max_element;
            }
        }

    public:
        void init( const index total_size ) {
            inserts_since_last_prune = 0;
            clear();
        }

	
	template<typename InputIterator>
	  void add_col(InputIterator begin, InputIterator end) {
	  for(InputIterator it=begin;it!=end;it++) {
	    data.push(*it);
	    inserts_since_last_prune++;
	  }
	  if( 2 * inserts_since_last_prune >( index ) data.size( ) )
	    prune();
	}
	
	
        void add_col( const column& col ) {
            for( index idx = 0; idx < (index) col.size(); idx++ )
                data.push( col[ idx ] );
            inserts_since_last_prune += col.size( );
            if( 2 * inserts_since_last_prune >( index ) data.size( ) )
                prune();
        }

        index get_max_index() {
            index max_element = pop_max_index( );
            if( max_element == -1 )
                return -1;
            else {
                data.push( max_element );
                return max_element;
            }
        }

        void get_col_and_clear( column& col ) {
            col.clear();
            index max_index = pop_max_index( );
            while( max_index != -1 ) {
                col.push_back( max_index );
                max_index = pop_max_index( );
            }
            std::reverse( col.begin(), col.end() );
        }

        bool is_empty() {
            return get_max_index() == -1;
        }

		void clear() {
            data = std::priority_queue< index >();
		}

		void remove_max() {
            pop_max_index();
        }

        void set_col( const column& col  ) {
            clear();
            add_col( col );
        }

        void get_col( column& col  ) {
            get_col_and_clear( col );
            add_col( col );
        }
    };

}
// END copied from multicritical/phat_mod/include/phat/representations/heap_pivot_column.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/bit_tree_pivot_column.h
/*  Copyright 2013 IST Austria
    Contributed by: Hubert Wagner

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {

    // This is a bitset indexed with a 64-ary tree. Each node in the index
    // has 64 bits; i-th bit says that the i-th subtree is non-empty.
    // Supports practically O(1), inplace, zero-allocation: insert, remove, max_element
    // and clear in O(number of ones in the bitset).
    // 'add_index' is still the real bottleneck in practice.
    class bit_tree_column
    {
    protected:

        size_t offset; // data[i + offset] = ith block of the data-bitset
        typedef uint64_t block_type;
        std::vector< block_type > data;

        
        size_t debrujin_magic_table[ 64 ];

        enum { block_size_in_bits = 64 };
        enum { block_shift = 6 };

        // Some magic: http://graphics.stanford.edu/~seander/bithacks.html
        // Gets the position of the rightmost bit of 'x'. 0 means the most significant bit. 
        // (-x)&x isolates the rightmost bit.
        // The whole method is much faster than calling log2i, and very comparable to using ScanBitForward/Reverse intrinsic,
        // which should be one CPU instruction, but is not portable.
        size_t rightmost_pos( const block_type value ) const {                
            return 64 - 1 - debrujin_magic_table[ ( (value & (-(int64_t)value) ) * 0x07EDD5E59A4E28C2 ) >> 58 ];
        }

    public:        

        void init( index num_cols ) {
            int64_t n = 1; // in case of overflow
            int64_t bottom_blocks_needed = ( num_cols + block_size_in_bits - 1 ) / block_size_in_bits;
            int64_t upper_blocks = 1;        

            // How many blocks/nodes of index needed to index the whole bitset?
            while( n * block_size_in_bits < bottom_blocks_needed ) {
                n *= block_size_in_bits;
                upper_blocks += n;
            }

            offset = upper_blocks;
            data.resize( upper_blocks + bottom_blocks_needed, 0 );

            std::size_t temp_array[ 64 ] = {
                    63,  0, 58,  1, 59, 47, 53,  2,
                    60, 39, 48, 27, 54, 33, 42,  3,
                    61, 51, 37, 40, 49, 18, 28, 20,
                    55, 30, 34, 11, 43, 14, 22,  4,
                    62, 57, 46, 52, 38, 26, 32, 41,
                    50, 36, 17, 19, 29, 10, 13, 21,
                    56, 45, 25, 31, 35, 16,  9, 12,
                    44, 24, 15,  8, 23,  7,  6,  5 };

            std::copy( &temp_array[ 0 ], &temp_array[ 64 ], &debrujin_magic_table[ 0 ] );
        }

        index get_max_index() const {
            if( !data[ 0 ] )
                return -1;

            size_t n = 0;
            size_t newn = 0;
            size_t index = 0;
            while( newn < data.size() ) {
                n = newn;
                index = rightmost_pos( data[ n ] );
                newn = ( n << block_shift ) + index + 1;
            } 

            return ( ( n - offset ) << block_shift ) + index;
        }

        bool is_empty() const {
            return data[ 0 ] == 0;
        }

        void add_index( const size_t entry ) {
            const block_type ONE = 1;
            const block_type block_modulo_mask = ( ONE << block_shift ) - 1;
            size_t index_in_level = entry >> block_shift;
            size_t address = index_in_level + offset;
            size_t index_in_block = entry & block_modulo_mask;

            block_type mask = ( ONE << ( block_size_in_bits - index_in_block - 1 ) );

            data[ address ] ^= mask;

            // Check if we reached the root. Also, if anyone else was in this block, we don't need to update the path up.
            while( address && !( data[ address ] & ~mask ) ) {
                index_in_block = index_in_level & block_modulo_mask;
                index_in_level >>= block_shift;
                --address;
                address >>= block_shift;
                mask = ( ONE << ( block_size_in_bits - index_in_block - 1 ) );
                data[ address ] ^= mask;
            }
        }

        void get_col_and_clear( column &out ) {
            index mx = this->get_max_index();
            while( mx != -1 ) {
                out.push_back( mx );
                add_index( mx );
                mx = this->get_max_index();
            }

            std::reverse( out.begin(), out.end() );
        }

	
	template<typename InputIterator>
	  void add_col(InputIterator begin, InputIterator end) {
	  for(InputIterator it=begin;it!=end;it++) {
	    add_index(*it);
	  }
	}
	

        void add_col(const column &col) {
            for( size_t i = 0; i < col.size(); ++i )
                add_index(col[i]);
        }

		void clear() {
            index mx = this->get_max_index();
            while( mx != -1 ) {
                add_index( mx );
                mx = this->get_max_index();
            }
        }

		void remove_max() {
            add_index( get_max_index() );
        }

        void set_col( const column& col ) {
            clear();
            add_col( col );
        }

        void get_col( column& col ) {
            get_col_and_clear( col );
            add_col( col );
        }
    };

}
// END copied from multicritical/phat_mod/include/phat/representations/bit_tree_pivot_column.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/vector_column_rep.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class vector_column_rep {

    protected:

        std::vector<index> indices;

        thread_local_storage< column >* temp_column_buffer;

    public:

	typedef vector_column_rep Self;

	void offer_thread_local_storage(thread_local_storage< column >* tls) {
	  temp_column_buffer=tls;
	}

	std::vector<index>::const_iterator begin() const {
	  return indices.begin();
	}

	std::vector<index>::const_iterator end() const {
	  return indices.end();
	}

        // replaces(!) content of 'col' with boundary of given index
        void _get_col( column& col  ) const {
	  col.clear();
	  col = indices; 
        }
        void _set_col( const column& col  ) { 
            indices = col; 
        }

	void _set_nr_of_rows( int nr_of_rows ) {
	  // ignore
	}

        // true iff boundary of given idx is empty
        bool _is_empty() const { 
            return indices.empty(); 
        }

        // largest row index of given column idx (new name for lowestOne())
        index _get_max_index() const { 
            return indices.empty() ? -1 : indices.back(); 
        }

        // removes the maximal index of a column
        void _remove_max() {
            indices.pop_back();
        }

        // clears given column
        void _clear() { 
            indices.clear();
        }

        // syncronizes all data structures (essential for openmp stuff)
        void _sync() {}

        // adds column 'source' to column 'target'
        void _add_to( const Self& source_col ) {
	  column& temp_col = (*temp_column_buffer)();
	              
	  size_t new_size = source_col.indices.size() + this->indices.size();
	  
	  if (new_size > temp_col.size()) temp_col.resize(new_size);
          
	  std::vector<index>::iterator col_end = std::set_symmetric_difference( this->indices.begin(), this->indices.end(),
										source_col.indices.begin(), source_col.indices.end(),
										temp_col.begin() );
	  temp_col.erase(col_end, temp_col.end());
	  
          
	  this->indices.swap(temp_col);
        }
        
        // finalizes given column
        void _finalize() {
            column(indices.begin(), indices.end()).swap(indices);
        }
    };
}
// END copied from multicritical/phat_mod/include/phat/representations/vector_column_rep.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/list_column_rep.h
/*  Copyright 2013 IST Austria
    Contributed by: Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class list_column_rep {

    protected:
      std::list< index > indices;

    public:

        typedef list_column_rep Self;
      
	std::list<index>::const_iterator begin() const {
	  return indices.begin();
	}

	std::list<index>::const_iterator end() const {
	  return indices.end();
	}

	void offer_thread_local_storage(thread_local_storage< column >* tls) {
	}

        // replaces(!) content of 'col' with boundary of given index
        void _get_col( column& col  ) const {
            col.clear();
            col.reserve( indices.size() );
            std::copy (indices.begin(), indices.end(), std::back_inserter(col) );
        }

        void _set_col( const column& col  ) {
            indices.clear();
            indices.resize( col.size() );
            std::copy (col.begin(), col.end(), indices.begin() );
        }

	void _set_nr_of_rows( int nr_of_rows ) {
	  // ignore
	}

        // true iff boundary of given idx is empty
        bool _is_empty() const {
            return indices.empty();    
        }

        // largest row index of given column idx (new name for lowestOne())
        index _get_max_index() const {
            return indices.empty() ? -1 : *indices.rbegin();
        }

        // removes the maximal index of a column
        void _remove_max() {
            std::list< index >::iterator it = indices.end();
            it--;
            indices.erase( it );
        }
        
        // clears given column
        void _clear() {
            indices.clear();    
        }

        // syncronizes all data structures (essential for openmp stuff)
        void _sync() {}

        // adds column 'source' to column 'target'
        void _add_to( Self& source) {
            std::list< index >& source_col = source.indices;
            std::list< index >& target_col = this->indices;
            std::list< index > temp_col;
            target_col.swap( temp_col );
            std::set_symmetric_difference( temp_col.begin(), temp_col.end(),
                                           source_col.begin(), source_col.end(),
                                           std::back_inserter( target_col ) );
        }
        
        // finalizes given column
        void _finalize() {
        }
    };
}
// END copied from multicritical/phat_mod/include/phat/representations/list_column_rep.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/set_column_rep.h
/*  Copyright 2013 IST Austria
    Contributed by: Ulrich Bauer, Michael Kerber, Jan Reininghaus

    This file is part of PHAT.

    PHAT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PHAT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class set_column_rep {

    protected:
      std::set< index > indices;

    public:

	typedef set_column_rep Self;

	std::set<index>::const_iterator begin() const {
	  return indices.begin();
	}

	std::set<index>::const_iterator end() const {
	  return indices.end();
	}

	void offer_thread_local_storage(thread_local_storage< column >* tls) {
	}

        // replaces(!) content of 'col' with boundary of given index
        void _get_col( column& col  ) const {
            col.clear();
            col.reserve( indices.size() );
            std::copy (indices.begin(), indices.end(), std::back_inserter(col) );
        }
        void _set_col( const column& col  ) {
            indices.clear();
            indices.insert( col.begin(), col.end() );
        }

	void _set_nr_of_rows( int nr_of_rows ) {
	  // ignore
	}

        // true iff boundary of given idx is empty
        bool _is_empty() const {
            return indices.empty();    
        }

        // largest row index of given column idx (new name for lowestOne())
        index _get_max_index() const {
            return indices.empty() ? -1 : *indices.rbegin();
        }

        // removes the maximal index of a column
        void _remove_max() {
            std::set< index >::iterator it = indices.end();
            it--;
            indices.erase( it );
        }
        
        // clears given column
        void _clear() {
            indices.clear();    
        }

        // syncronizes all data structures (essential for openmp stuff)
        void _sync() {}

        // adds column 'source' to column 'target'
        void _add_to( Self& source) {

   	    std::set< index >& col = this->indices;
            for( std::set< index >::iterator it = source.indices.begin(); it != source.indices.end(); it++ ) {
				std::pair< std::set< index >::iterator, bool > result = col.insert( *it );
				if( !result.second ) 
					col.erase( result.first );
			}
        }
        
        // finalizes given column
        void _finalize() {
        }

    };
}
// END copied from multicritical/phat_mod/include/phat/representations/set_column_rep.h
// BEGIN copied from multicritical/phat_mod/include/phat/representations/heap_column_rep.h
/*  Copyright 2013 IST Austria
Contributed by: Jan Reininghaus

This file is part of PHAT.

PHAT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PHAT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with PHAT.  If not, see <http://www.gnu.org/licenses/>. */



namespace phat {
    class heap_column_rep {

    protected:
        std::vector< index > indices;

        index inserts_since_last_prune;

        mutable thread_local_storage< column >* temp_column_buffer;

    protected:
        void _prune()
        {
            column& col = indices;
            column& temp_col = (*temp_column_buffer)();
            temp_col.clear();
            index max_index = _pop_max_index( col );
            while( max_index != -1 ) {
                temp_col.push_back( max_index );
                max_index = _pop_max_index( col );
            }
            col = temp_col;
            std::reverse( col.begin( ), col.end( ) );
            std::make_heap( col.begin( ), col.end( ) );
            inserts_since_last_prune = 0;
        }

        index _pop_max_index(column& col) const
        {
            if( col.empty( ) )
                return -1;
            else {
                index max_element = col.front( );
                std::pop_heap( col.begin( ), col.end( ) );
                col.pop_back( );
                while( !col.empty( ) && col.front( ) == max_element ) {
                    std::pop_heap( col.begin( ), col.end( ) );
                    col.pop_back( );
                    if( col.empty( ) )
                        return -1;
                    else {
                        max_element = col.front( );
                        std::pop_heap( col.begin( ), col.end( ) );
                        col.pop_back( );
                    }
                }
                return max_element;
            }
        }

    public:

	typedef heap_column_rep Self;

	std::vector<index>::const_iterator begin() const {
	  return indices.begin();
	}

	std::vector<index>::const_iterator end() const {
	  return indices.end();
	}

	void offer_thread_local_storage(thread_local_storage< column >* tls) {
	  temp_column_buffer=tls;
	}

        // replaces(!) content of 'col' with boundary of given index
        void _get_col( column& col ) const
        {
	  col.clear();
	  (*temp_column_buffer)() = indices;
	  index max_index = _pop_max_index( (*temp_column_buffer)() );
	  while( max_index != -1 ) {
	    col.push_back( max_index );
	    max_index = _pop_max_index( (*temp_column_buffer)() );
	  }
	  std::reverse( col.begin( ), col.end( ) );
        }
        void _set_col(const column& col )
        {
            indices = col;
            std::make_heap( indices.begin( ), indices.end( ) );
        }

	void _set_nr_of_rows( int nr_of_rows ) {
	  // ignore
	}

        // true iff boundary of given idx is empty
        bool _is_empty() const
        {
            return _get_max_index() == -1;
        }

        // largest row index of given column idx (new name for lowestOne())
        index _get_max_index() const
        {
            column& col = const_cast< column& >( indices );
            index max_element = _pop_max_index( col );
            col.push_back( max_element );
            std::push_heap( col.begin( ), col.end( ) );
            return max_element;
        }

        // removes the maximal index of a column
        void _remove_max()
        {
            _pop_max_index( indices );
        }

        // clears given column
        void _clear()
        {
            indices.clear( );
        }

	index _size() {
	    _prune();
	    return indices.size();
	}

        // syncronizes all data structures (essential for openmp stuff)
        void _sync( ) {}

        // adds column 'source' to column 'target'
        void _add_to( const Self& source )
        {           
            for( index idx = 0; idx < (index)source.indices.size( ); idx++ ) {
                this->indices.push_back( source.indices[ idx ] );
                std::push_heap( this->indices.begin(), this->indices.end() );
            }
            inserts_since_last_prune += source.indices.size();

            if( 2 * inserts_since_last_prune > ( index )this->indices.size() )
                _prune();
        }
        
        // finalizes given column
        void _finalize() {
            _prune();
        }

    };
}
// END copied from multicritical/phat_mod/include/phat/representations/heap_column_rep.h


namespace phat {

 // typedef Uniform_representation< std::unordered_map<index,phat::vector_column_rep>, std::unordered_map<index,index> > hash_vector;
 typedef Uniform_representation< std::vector<phat::vector_column_rep>, std::vector<index> > vector_vector;
 typedef Uniform_representation< std::vector<phat::set_column_rep>, std::vector<index> >    vector_set;
 typedef Uniform_representation< std::vector<phat::heap_column_rep>, std::vector<index> >   vector_heap;
 typedef Uniform_representation< std::vector<phat::list_column_rep>, std::vector<index> >   vector_list;
 typedef Pivot_representation< vector_vector, full_column >     full_pivot_column;
 typedef Pivot_representation< vector_vector, heap_column >     heap_pivot_column;
 typedef Pivot_representation< vector_vector, sparse_column >   sparse_pivot_column;
 typedef Pivot_representation< vector_vector, bit_tree_column > bit_tree_pivot_column;

}
// END copied from multicritical/phat_mod/include/phat/representations/default_representations.h

namespace phat {

// interface class for the main data structure -- implementations of the interface can be found in ./representations

 template< class Representation = bit_tree_pivot_column >
   class boundary_matrix
    {
        
    protected:
        Representation rep;

    // interface functions -- actual implementation and complexity depends on chosen @Representation template
    public:
        // get overall number of columns in boundary_matrix
        index get_num_cols() const { return rep._get_num_cols(); }

        // sets the number of rows (1st parameter) and columns (2nd parameter)
        // of the matrix. Most internal types ignore the number of rows
        // but some do no
        void set_dimensions( index nr_of_rows, index nr_of_columns ) {rep._set_dimensions( nr_of_rows, nr_of_columns );}

        // set overall number of columns in boundary_matrix
        // sets the number of rows to nr_of_columns as well!
        void set_num_cols( index nr_of_columns ) { rep._set_dimensions( nr_of_columns, nr_of_columns ); }

        // get dimension of given index
        dimension get_dim( index idx ) const { return rep._get_dim( idx ); }

        // set dimension of given index
        void set_dim( index idx, dimension dim ) { rep._set_dim( idx, dim ); }

        // replaces content of @col with boundary of given index
        void get_col( index idx, column& col  ) const { 
	  rep._get_col( idx, col ); 
	}

        // set column @idx to the values contained in @col
        void set_col( index idx, const column& col  ) { 
	  rep._set_col( idx, col ); }

        // true iff boundary of given column is empty
        bool is_empty( index idx ) const { 
	  return rep._is_empty(idx); 
	}

        // largest index of given column (new name for lowestOne()) -- NOT thread-safe
        index get_max_index( index idx ) const { 
	  return rep._get_max_index(idx); }

        // removes maximal index from given column 
        void remove_max( index idx ) { 
	  return rep._remove_max( idx ); 
	}

        // adds column @source to column @target'
        void add_to( index source, index target ) { 
	  rep._add_to( source, target ); }

        // clears given column
        void clear( index idx ) { 
	  rep._clear(idx); }
        
        // finalizes given column
        void finalize( index idx ) { 
	  rep._finalize(idx); }
        
        // synchronizes all internal data structures -- has to be called before and after any multithreaded access!
        void sync() { rep._sync(); }

    // info functions -- independent of chosen 'Representation'
    public:
        // maximal dimension
        dimension get_max_dim() const {
            dimension cur_max_dim = 0;
            for( index idx = 0; idx < get_num_cols(); idx++ )
                cur_max_dim = get_dim( idx ) > cur_max_dim ? get_dim( idx ) : cur_max_dim;
            return cur_max_dim;
        }

        // number of nonzero rows for given column @idx
        index get_num_rows( index idx ) const {
            column cur_col;
            get_col( idx, cur_col );
            return cur_col.size();
        }

        // maximal number of nonzero rows of all columns
        index get_max_col_entries() const {
            index max_col_entries = -1;
            const index nr_of_columns = get_num_cols();
            for( index idx = 0; idx < nr_of_columns; idx++ )
                max_col_entries = get_num_rows( idx ) > max_col_entries ? get_num_rows( idx ) : max_col_entries;
            return max_col_entries;
        }

        // maximal number of nonzero cols of all rows
        index get_max_row_entries() const {
            size_t max_row_entries = 0;
            const index nr_of_columns = get_num_cols();
            std::vector< std::vector< index > > transposed_matrix( nr_of_columns );
            column temp_col;
            for( index cur_col = 0; cur_col < nr_of_columns; cur_col++ ) {
                get_col( cur_col, temp_col );
                for( index idx = 0; idx < (index)temp_col.size(); idx++)
                    transposed_matrix[ temp_col[ idx ]  ].push_back( cur_col );
            }
            for( index idx = 0; idx < nr_of_columns; idx++ )
                max_row_entries = transposed_matrix[ idx ].size() > max_row_entries ? transposed_matrix[ idx ].size() : max_row_entries;
            return max_row_entries;
        }

        // overall number of entries in the matrix
        index get_num_entries() const {
            index number_of_nonzero_entries = 0;
            const index nr_of_columns = get_num_cols();
            for( index idx = 0; idx < nr_of_columns; idx++ )
                number_of_nonzero_entries += get_num_rows( idx );
            return number_of_nonzero_entries;
        }
    
    // operators / constructors
    public:
        boundary_matrix() {};

        template< class OtherRepresentation >
        boundary_matrix( const boundary_matrix< OtherRepresentation >& other ) {
            *this = other;
        }

        boundary_matrix( const boundary_matrix<Representation>& other) {
	  *this = other;
	}

        template< typename OtherRepresentation >
        bool operator==( const boundary_matrix< OtherRepresentation >& other_boundary_matrix ) const {
            const index number_of_columns = this->get_num_cols();

            if( number_of_columns != other_boundary_matrix.get_num_cols() )
                return false;

            column temp_col;
            column other_temp_col;
            for( index idx = 0; idx < number_of_columns; idx++ ) {
                this->get_col( idx, temp_col );
                other_boundary_matrix.get_col( idx, other_temp_col );
                if( temp_col != other_temp_col || this->get_dim( idx ) != other_boundary_matrix.get_dim( idx ) )
                    return false;
            }
            return true;
        }

        template< typename OtherRepresentation >
        bool operator!=( const boundary_matrix< OtherRepresentation >& other_boundary_matrix ) const {
            return !( *this == other_boundary_matrix );
        }


        template< typename OtherRepresentation >
        boundary_matrix< Representation >& assign(const boundary_matrix< OtherRepresentation >& other) {
	  const index nr_of_columns = other.get_num_cols();
	  this->set_num_cols( nr_of_columns );
	  column temp_col;
	  for( index cur_col = 0; cur_col <  nr_of_columns; cur_col++ ) {
	    this->set_dim( cur_col, other.get_dim( cur_col ) );
	    other.get_col( cur_col, temp_col );
	    this->set_col( cur_col, temp_col );
	  }
	  // by convention, always return *this
	  return *this;
	}

        boundary_matrix< Representation >& operator=( const boundary_matrix< Representation >& other )
        {
	  return assign(other);
        }

        template< typename OtherRepresentation >
        boundary_matrix< Representation >& operator=( const boundary_matrix< OtherRepresentation >& other )
        {
	  return assign(other);
        }

    // I/O -- independent of chosen 'Representation'
    public:

        // initializes boundary_matrix from (vector<vector>, vector) pair -- untested
        template< typename index_type, typename dimemsion_type >
        void load_vector_vector( const std::vector< std::vector< index_type > >& input_matrix, const std::vector< dimemsion_type >& input_dims ) { 
            const index nr_of_columns = (index)input_matrix.size();
            this->set_num_cols( nr_of_columns );
            column temp_col;
            #pragma omp parallel for private( temp_col )
            for( index cur_col = 0; cur_col <  nr_of_columns; cur_col++ ) {
	      //std::cout << "At index " << cur_col  << " of " << nr_of_columns << " " << omp_get_thread_num() << std::endl;
                this->set_dim( cur_col, (dimension)input_dims[ cur_col ] );
                
                index num_rows = input_matrix[ cur_col ].size();
                temp_col.resize( num_rows );
                for( index cur_row = 0; cur_row <  num_rows; cur_row++ )
                    temp_col[ cur_row ] = (index)input_matrix[ cur_col ][ cur_row ];
                this->set_col( cur_col, temp_col );
            }
        }

        template< typename index_type, typename dimemsion_type >
        void save_vector_vector( std::vector< std::vector< index_type > >& output_matrix, std::vector< dimemsion_type >& output_dims ) { 
            const index nr_of_columns = get_num_cols();
            output_matrix.resize( nr_of_columns );
            output_dims.resize( nr_of_columns );
            column temp_col;
            for( index cur_col = 0; cur_col <  nr_of_columns; cur_col++ ) {
                output_dims[ cur_col ] = (dimemsion_type)get_dim( cur_col );
                get_col( cur_col, temp_col );
                index num_rows = temp_col.size();
                output_matrix[ cur_col ].clear();
                output_matrix[ cur_col ].resize( num_rows );
                for( index cur_row = 0; cur_row <  num_rows; cur_row++ )
                    output_matrix[ cur_col ][ cur_row ] = (index_type)temp_col[ cur_row ];
            }
        }


        // Loads the boundary_matrix from given file in ascii format 
        // Format: each line represents a column, first number is dimension, other numbers are the content of the column.
        // Ignores empty lines and lines starting with a '#'.
        bool load_ascii( std::string filename ) { 
            // first count number of columns:
            std::string cur_line;
            std::ifstream dummy( filename .c_str() );
            if( dummy.fail() )
                return false;

            index number_of_columns = 0;
            while( getline( dummy, cur_line ) ) {
                cur_line.erase(cur_line.find_last_not_of(" \t\n\r\f\v") + 1);
                if( cur_line != "" && cur_line[ 0 ] != '#' )
                    number_of_columns++;

            }
            this->set_num_cols( number_of_columns );
            dummy.close();

            std::ifstream input_stream( filename.c_str() );
            if( input_stream.fail() )
                return false;
            
            column temp_col;
            index cur_col = -1;
            while( getline( input_stream, cur_line ) ) {
                cur_line.erase(cur_line.find_last_not_of(" \t\n\r\f\v") + 1);
                if( cur_line != "" && cur_line[ 0 ] != '#' ) {
                    cur_col++;
                    std::stringstream ss( cur_line );
                    
                    int64_t temp_dim;
                    ss >> temp_dim;
                    this->set_dim( cur_col, (dimension) temp_dim );

                    int64_t temp_index;
                    temp_col.clear();
                    while( ss.good() ) {
                        ss >> temp_index;
                        temp_col.push_back( (index)temp_index );
                    }
                    std::sort( temp_col.begin(), temp_col.end() );
                    this->set_col( cur_col, temp_col );
                }
            }

            input_stream.close();
            return true;
        }

        // Saves the boundary_matrix to given file in ascii format
        // Format: each line represents a column, first number is dimension, other numbers are the content of the column
        bool save_ascii( std::string filename ) {
            std::ofstream output_stream( filename.c_str() );
            if( output_stream.fail() )
                return false;

            const index nr_columns = this->get_num_cols();
            column tempCol;
            for( index cur_col = 0; cur_col < nr_columns; cur_col++ ) {
                output_stream << (int64_t)this->get_dim( cur_col );
                this->get_col( cur_col, tempCol );
                for( index cur_row_idx = 0; cur_row_idx < (index)tempCol.size(); cur_row_idx++ )
                    output_stream << " " << tempCol[ cur_row_idx ];
                output_stream << std::endl;
            }

            output_stream.close();
            return true;
        }

        // Loads boundary_matrix from given file 
        // Format: nr_columns % dim1 % N1 % row1 row2 % ...% rowN1 % dim2 % N2 % ...
        bool load_binary(const std::string &filename )
        {
            std::ifstream input_stream( filename.c_str( ), std::ios_base::binary | std::ios_base::in );
            if( input_stream.fail( ) )
                return false;

            //TODO Note that if you read a file that isn't actually a data file, you may get
            //a number of columns that is bigger than the available memory, which leads to crashes
            //with deeply confusing error messages. Consider ways to prevent this.
            //Magic number in the file header? Check for more columns than bytes in the file?
            int64_t nr_columns;
            input_stream.read( (char*)&nr_columns, sizeof( int64_t ) );
            this->set_num_cols( (index)nr_columns );

            column temp_col;
            for( index cur_col = 0; cur_col < nr_columns; cur_col++ ) {
                int64_t cur_dim;
                input_stream.read( (char*)&cur_dim, sizeof( int64_t ) );
                this->set_dim( cur_col, (dimension)cur_dim );
                int64_t nr_rows;
                input_stream.read( (char*)&nr_rows, sizeof( int64_t ) );
                temp_col.resize( ( std::size_t )nr_rows );
                for( index idx = 0; idx < nr_rows; idx++ ) {
                    int64_t cur_row;
                    input_stream.read( (char*)&cur_row, sizeof( int64_t ) );
                    temp_col[ idx ] = (index)cur_row;
                }
                this->set_col( cur_col, temp_col );
            }

            input_stream.close( );
            return true;
        }

        // Saves the boundary_matrix to given file in binary format 
        // Format: nr_columns % dim1 % N1 % row1 row2 % ...% rowN1 % dim2 % N2 % ...
        bool save_binary( std::string filename )
        {
            std::ofstream output_stream( filename.c_str( ), std::ios_base::binary | std::ios_base::out );
            if( output_stream.fail( ) )
                return false;

            const int64_t nr_columns = this->get_num_cols( );
            output_stream.write( (char*)&nr_columns, sizeof( int64_t ) );
            column tempCol;
            for( index cur_col = 0; cur_col < nr_columns; cur_col++ ) {
                int64_t cur_dim = this->get_dim( cur_col );
                output_stream.write( (char*)&cur_dim, sizeof( int64_t ) );
                this->get_col( cur_col, tempCol );
                int64_t cur_nr_rows = tempCol.size( );
                output_stream.write( (char*)&cur_nr_rows, sizeof( int64_t ) );
                for( index cur_row_idx = 0; cur_row_idx < (index)tempCol.size( ); cur_row_idx++ ) {
                    int64_t cur_row = tempCol[ cur_row_idx ];
                    output_stream.write( (char*)&cur_row, sizeof( int64_t ) );
                }
            }

            output_stream.close( );
            return true;
        }
    };
}
// END copied from multicritical/phat_mod/include/phat/boundary_matrix.h

