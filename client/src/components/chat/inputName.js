import React from 'react';
import Autosuggest from 'react-autosuggest';
import match from 'autosuggest-highlight/match';
import parse from 'autosuggest-highlight/parse';
import { withStyles } from '@material-ui/core/styles';
import TextField from '../uielements/textfield';
import Paper from '../uielements/paper';
import { MenuItem } from './inputName.style';

const renderInput = inputProps => {
  const { classes, autoFocus, value, ref, ...other } = inputProps;

  return (
    <TextField
      autoFocus={autoFocus}
      className={classes.textField}
      value={value}
      inputRef={ref}
      InputProps={{
        classes: {
          input: classes.input,
        },
        ...other,
      }}
    />
  );
};

const renderSuggestionsContainer = options => {
  const { containerProps, children } = options;

  return (
    <Paper {...containerProps} square>
      {children}
    </Paper>
  );
};

const getSuggestionValue = user => {
  return user.name;
};

const getSuggestions = (value, users) => {
  const inputValue = value.trim().toLowerCase();
  const inputLength = inputValue.length;
  let count = 0;

  return inputLength === 0
    ? []
    : users.filter(user => {
        const keep =
          count < 5 &&
          user.name.toLowerCase().slice(0, inputLength) === inputValue;

        if (keep) {
          count += 1;
        }

        return keep;
      });
};

const styles = theme => ({
  container: {
    flexGrow: 1,
    position: 'relative',
  },
  suggestionsContainerOpen: {
    position: 'absolute',
    marginTop: theme.spacing(1),
    marginBottom: theme.spacing(3),
    zIndex: 1501,
    left: 0,
    right: 0,
  },
  suggestion: {
    display: 'block',
  },
  suggestionsList: {
    margin: 0,
    padding: 0,
    listStyleType: 'none',
  },
  textField: {
    width: '100%',
  },
});

class InputName extends React.Component {
  state = {
    value: '',
    suggestions: [],
  };

  handleSuggestionsFetchRequested = ({ value }) => {
    const { users } = this.props;
    this.setState({
      suggestions: getSuggestions(value, users),
    });
  };

  handleSuggestionsClearRequested = () => {
    this.setState({
      suggestions: [],
    });
  };

  handleChange = (event, { newValue }) => {
    this.setState({ value: newValue });
  };
  renderSuggestion = (user, { query, isHighlighted }) => {
    const matches = match(user.name, query);
    const parts = parse(user.name, matches);
    const onClick = () => {
      this.props.setComposedId(user.id);
    };
    return (
      <MenuItem selected={isHighlighted} component="div" onClick={onClick}>
        <div className="userSuggestion">
          <div className="userImg">
            <img alt="#" src={user.profileImageUrl} />
          </div>

          <p className="suggetionText">
            {parts.map((part, index) => {
              return part.highlight ? (
                <strong key={index} style={{ fontWeight: 500 }}>
                  {part.text}
                </strong>
              ) : (
                <span key={index} style={{ fontWeight: 300 }}>
                  {part.text}
                </span>
              );
            })}
          </p>
        </div>
      </MenuItem>
    );
  };
  render() {
    const { classes } = this.props;
    return (
      <Autosuggest
        theme={{
          container: classes.container,
          suggestionsContainerOpen: classes.suggestionsContainerOpen,
          suggestionsList: classes.suggestionsList,
          suggestion: classes.suggestion,
        }}
        renderInputComponent={renderInput}
        suggestions={this.state.suggestions}
        onSuggestionsFetchRequested={this.handleSuggestionsFetchRequested}
        onSuggestionsClearRequested={this.handleSuggestionsClearRequested}
        renderSuggestionsContainer={renderSuggestionsContainer}
        getSuggestionValue={getSuggestionValue}
        renderSuggestion={this.renderSuggestion}
        className={this.props.className}
        inputProps={{
          autoFocus: true,
          classes,
          placeholder: 'Search a user',
          value: this.state.value,
          onChange: this.handleChange,
        }}
      />
    );
  }
}

export default withStyles(styles)(InputName);
