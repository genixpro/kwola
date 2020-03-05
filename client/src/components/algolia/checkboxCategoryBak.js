import React from 'react';
import { connectRefinementList } from 'react-instantsearch/connectors';
import Checkbox from '../uielements/checkbox';
import List, { ListItem, ListItemText } from '../uielements/lists';

const CheckBoxItem = ({ item, refine }) => (
  <ListItem
    onClick={e => {
      e.preventDefault();
      refine(item.value);
    }}
  >
    <Checkbox checked={Boolean(item.isRefined)} />
    <ListItemText>{item.label}</ListItemText>
  </ListItem>
);

const MaterialUiCheckBoxRefinementList = ({
  items,
  attributeName,
  refine,
  createURL,
}) => (
  <List>
    {/* <span style={{ fontSize: 18 }}>{attributeName.toUpperCase()}</span> */}
    {items.map(item => (
      <CheckBoxItem
        key={item.label}
        item={item}
        refine={refine}
        createURL={createURL}
      />
    ))}
  </List>
);
const ConnectedCheckBoxRefinementList = connectRefinementList(
  MaterialUiCheckBoxRefinementList
);

export default () => (
  <ConnectedCheckBoxRefinementList attribute="categories" operator="or" />
);
