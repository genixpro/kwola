import React from 'react';
import { connectSearchBox } from 'react-instantsearch/connectors';
import {
  TextField,
  ClearIcon,
  ClearIconButtton,
  SearchBoxWrapper
} from './algoliaComponent.style';
const MaterialUiSearchBox = ({ currentRefinement, refine }) => {
  const clear = currentRefinement ? (
    <ClearIconButtton color="primary" onClick={() => refine('')}>
      <ClearIcon>clear</ClearIcon>
    </ClearIconButtton>
  ) : null;
  return (
    <SearchBoxWrapper>
      <TextField
        id="SearchBox"
        value={currentRefinement}
        onChange={e => refine(e.target.value)}
        label="Search for a product..."
      />
      {clear}
    </SearchBoxWrapper>
  );
};
export default connectSearchBox(MaterialUiSearchBox);
