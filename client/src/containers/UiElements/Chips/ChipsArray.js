import React from 'react';
import { ArrayChips, Wrapper } from './chips.style';
class ChipsArray extends React.Component {
  state = {
    chipData: [
      { key: 0, label: 'Angular' },
      { key: 1, label: 'JQuery' },
      { key: 2, label: 'Polymer' },
      { key: 3, label: 'ReactJS' },
      { key: 4, label: 'Vue.js' },
    ],
  };

  styles = {
    chip: {
      margin: 4,
    },
    wrapper: {
      display: 'flex',
      flexWrap: 'wrap',
    },
  };

  handleRequestDelete = data => () => {
    if (data.label === 'ReactJS') {
      alert('Why would you want to delete React?! :)'); // eslint-disable-line no-alert
      return;
    }

    const chipData = [...this.state.chipData];
    const chipToDelete = chipData.indexOf(data);
    chipData.splice(chipToDelete, 1);
    this.setState({ chipData });
  };

  render() {
    return (
      <Wrapper>
        {this.state.chipData.map(data => {
          return (
            <ArrayChips
              label={data.label}
              key={data.key}
              onDelete={this.handleRequestDelete(data)}
            />
          );
        })}
      </Wrapper>
    );
  }
}
export default ChipsArray;
