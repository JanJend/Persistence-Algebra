/**
 * @file mod_modules.hpp
 * @brief Aggregate include for persistence modules, submodules, and maps.
 *
 * The implementation is split into focused headers so clients may include
 * only what they use.  This replaces the former incomplete Interval_module
 * stub without removing any presentation-matrix API.
 */
#pragma once

#include <grlina/proj_chain_complex.hpp>
#include <grlina/mod_class.hpp>
#include <grlina/mod_submodule.hpp>
#include <grlina/mod_transformations.hpp>
#include <grlina/hom_interface.hpp>
#include <grlina/mod_constructions.hpp>
#include <grlina/mod_epsilon.hpp>
#include <grlina/mod_presentations.hpp>
#include <grlina/mod_landscape.hpp>
