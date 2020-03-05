import React from 'react';
import { connectRefinementList } from 'react-instantsearch/connectors';
import Checkbox from '../uielements/checkbox';
import { FormGroup, FormControlLabel } from '../uielements/form/';

const CheckBoxItem = ({ item, refine }) => (
  <FormControlLabel
    onClick={e => {
      e.preventDefault();
      refine(item.value);
    }}
    control={<Checkbox checked={Boolean(item.isRefined)} />}
    label={item.label}
  />
);

const MaterialUiCheckBoxRefinementList = ({
  items,
  attributeName,
  refine,
  createURL,
}) => (
  <FormGroup>
    {/* <span style={{ fontSize: 18 }}>{attributeName.toUpperCase()}</span> */}
    {items.map(item => (
      <CheckBoxItem
        key={item.label}
        item={item}
        refine={refine}
        createURL={createURL}
      />
    ))}
  </FormGroup>
);
const ConnectedCheckBoxRefinementList = connectRefinementList(
  MaterialUiCheckBoxRefinementList
);

export default () => (
  <ConnectedCheckBoxRefinementList attribute="categories" operator="or" />
);
