import React from 'react';
import { ListIcon, GridIcon, ViewChanger } from './algoliaComponent.style';

export default ({ view, changeView }) => (
  <ViewChanger>
    <button
      type="button"
      className={view === 'gridView' ? 'gridView active' : 'gridView'}
      onClick={() => changeView('gridView')}
    >
      <GridIcon>view_module</GridIcon>
    </button>
    <button
      type="button"
      className={view === 'gridView' ? 'listView' : 'listView active'}
      onClick={() => changeView('listView')}
    >
      <ListIcon>view_list</ListIcon>
    </button>
  </ViewChanger>
);
