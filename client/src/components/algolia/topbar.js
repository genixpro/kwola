import React from 'react';
import { Stats } from 'react-instantsearch/dom';
import SortBy from './sortby';
import ViewChanger from './viewChanger';
import { TopbarWrapper } from './algoliaComponent.style';

export default props => (
  <TopbarWrapper>
    <Stats {...props} />
    <div className="sortingOpt">
      <SortBy {...props} />
      <ViewChanger {...props} />
    </div>
  </TopbarWrapper>
);
