import React, { Component } from 'react';
import Input from '../components/uielements/input';
import Button, { Fab } from '../components/uielements/button';
import Icon from './uielements/icon';

export default class EditableComponent extends Component {
  constructor(props) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
    this.check = this.check.bind(this);
    this.edit = this.edit.bind(this);
    this.state = {
      value: this.props.value,
      editable: false,
    };
  }
  handleChange(event) {
    const value = event.target.value;
    this.setState({ value });
  }
  check() {
    this.setState({ editable: false });
    if (this.props.onChange) {
      this.props.onChange(this.props.itemKey, this.state.value);
    }
  }
  edit() {
    this.setState({ editable: true });
  }

  render() {
    const { value, editable } = this.state;
    return (
      <div className="matNoteContent">
        {editable ? (
          <div className="matNoteEditWrapper">
            <Input
              placeholder="Styled Hint Text"
              fullWidth
              name="text"
              value={value}
              className="matTodoInput"
              onChange={this.handleChange}
            />
            <Button
              variant="text"
              mini
              color="primary"
              type="check"
              className="matNoteEditIcon"
              onClick={this.check}
            >
              <Icon>done</Icon>
            </Button>
          </div>
        ) : (
          <p className="matNoteTextWrapper" onClick={this.edit}>
            {value || ' '}
            <Fab
              mini
              color="primary"
              aria-label="edit"
              type="edit"
              className="matNoteEditIcon"
            >
              <Icon>mode_edit</Icon>
            </Fab>
          </p>
        )}
      </div>
    );
  }
}
