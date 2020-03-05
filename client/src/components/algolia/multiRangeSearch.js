import React from 'react';
import { connectRange } from 'react-instantsearch/connectors';
import { Radio, RadioGroup } from './algoliaComponent.style';
import { FormControlLabel } from '../uielements/form';

const multiRangeSearch = ({ min, max, currentRefinement, refine }) => {
  const options = [
    { min, max: 10, value: ':10', label: '<$10' },
    { min: 10, max: 100, value: '10:100', label: '$10-$100' },
    { min: 100, max: 500, value: '100:500', label: '$100-$500' },
    { min: 500, max, value: '500:', label: '>$500' },
    { min, max, value: '', label: 'All' }
  ];
  const index = options.findIndex(
    option =>
      option.min === currentRefinement.min &&
      option.max === currentRefinement.max
  );
  const value = index > -1 ? options[index].value : '';

  return (
    <RadioGroup
      value={value}
      onChange={(event, price) => {
        let min = '',
          max = '';
        if (price) {
          const index = options.findIndex(option => option.value === price);
          if (index > -1) {
            min = options[index].min;
            max = options[index].max;
          }
        }
        refine({ min, max });
      }}
    >
      {options.map((option, index) => (
        <FormControlLabel
          key={index}
          value={option.value}
          control={<Radio color="primary" />}
          label={option.label}
        />
      ))}
    </RadioGroup>
  );
};
export default connectRange(multiRangeSearch);
