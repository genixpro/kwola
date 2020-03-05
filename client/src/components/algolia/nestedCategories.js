import React from 'react';
import { connectHierarchicalMenu } from 'react-instantsearch/connectors';
import List, { ListItem, ListItemText } from '../uielements/lists';

const NestedList = ({ id, items, refine }) => (
  <List>
    <span style={{ fontSize: 18 }}>{id.toUpperCase()}</span>
    {items.map((item, idx) => {
      const nestedElements = item.items
        ? item.items.map((child, childIdx) => (
            <ListItem
              key={childIdx}
              onClick={e => {
                e.preventDefault();
                refine(child.value);
              }}
              style={child.isRefined ? { fontWeight: 700 } : {}}
            >
              <ListItemText>{child.label}</ListItemText>
            </ListItem>
          ))
        : [];
      return (
        <ListItem
          key={idx}
          onClick={e => {
            e.preventDefault();
            refine(item.value);
          }}
          style={item.isRefined ? { fontWeight: 700 } : {}}
        >
          <ListItemText>{item.label}</ListItemText>
          <List>{nestedElements}</List>
        </ListItem>
      );
    })}
  </List>
);

const ConnectedNestedList = connectHierarchicalMenu(NestedList);

export default () => (
  <ConnectedNestedList
    id="Hierarchical Categories"
    attributes={[
      'hierarchicalCategories.lvl0',
      'hierarchicalCategories.lvl1',
      'hierarchicalCategories.lvl2'
    ]}
  />
);
