import React, { useState } from 'react';
import Icon from '@material-ui/core/Icon';
import TextField from '@material-ui/core/TextField';
import { SearchWrapper, ClearButton } from './SearchInput.style';

export default function SearchInput({ onChange = console.log }) {
  const [searchData, setSearchData] = useState('');

  const handleSearch = event => {
    setSearchData(event.target.value);
    onChange(event.target.value);
  };

  return (
    <SearchWrapper>
      <Icon>search</Icon>
      <TextField
        id="input-with-icon-grid"
        label="Search"
        value={searchData}
        onChange={handleSearch}
      />
      {searchData && (
        <ClearButton onClick={() => setSearchData('')}>
          <Icon>clear</Icon>
        </ClearButton>
      )}
    </SearchWrapper>
  );
}
