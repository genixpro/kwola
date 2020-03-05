import React from 'react';
import { RatingMenu } from 'react-instantsearch/dom';
import SearchText from './searchText';
import VoiceRecognition from './voiceRecognition';
import CheckboxCategory from './checkboxCategory';
// import RangeSlider from './rangeSlider';
// import NestedCategories from './nestedCategories';
import MultiRangeSearch from './multiRangeSearch';
import ClearAll from './clearAll';
import { SidebarWrapper, SidebarItem } from './algoliaComponent.style';

const Sidebar = props => (
  <SidebarWrapper className="algoliaSidebar">
    <SidebarItem className="contentBox">
      <SearchText {...props} />
    </SidebarItem>
    <SidebarItem className="contentBox">
      <h3 className="algoliaSidebarTitle">Voice Search</h3>
      <VoiceRecognition {...props} />
    </SidebarItem>
    <SidebarItem className="contentBox">
      <h3 className="algoliaSidebarTitle">Multi Range</h3>
      <MultiRangeSearch {...props} attribute="price" />
    </SidebarItem>
    {/* <SidebarItem className="contentBox">
      <h3 className="algoliaSidebarTitle" style={{ marginBottom: 10 }}>
        Slider
      </h3>
      <RangeSlider {...props} attribute="price" />
    </SidebarItem> */}
    <SidebarItem className="contentBox">
      <h3 className="algoliaSidebarTitle">Category</h3>
      <CheckboxCategory {...props} />
    </SidebarItem>
    {/* <SidebarItem className="contentBox">
      <NestedCategories {...props} />
    </SidebarItem> */}
    <SidebarItem className="contentBox">
      <h3 className="algoliaSidebarTitle">Rating</h3>
      <RatingMenu attribute="rating" max={5} />
    </SidebarItem>
    <SidebarItem className="contentBox">
      <ClearAll {...props} />
    </SidebarItem>
  </SidebarWrapper>
);

export default Sidebar;
