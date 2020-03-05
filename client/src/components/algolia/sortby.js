import React from 'react';
import { connectSortBy } from 'react-instantsearch/connectors';
import { MenuItem } from '../uielements/menus';
import { Select } from './algoliaComponent.style';

const MaterialUiSortBy = ({ items, currentRefinement, refine }) => (
  <Select
    value={currentRefinement}
    name="Sort by"
    onChange={event => refine(event.target.value)}
  >
    {items.map((item, index) => (
      <MenuItem key={index} value={item.value}>
        {item.label}
      </MenuItem>
    ))}
  </Select>
);

const ConnectedSortBy = connectSortBy(MaterialUiSortBy);

export default () => (
  <ConnectedSortBy
    items={[
      { value: 'default_search', label: 'Default' },
      { value: 'price_asc', label: 'Lowest Price' },
      { value: 'price_desc', label: 'Highest Price' }
    ]}
    defaultRefinement="default_search"
  />
);
