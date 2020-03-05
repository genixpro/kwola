import React, { Component } from 'react';
import MailComposeBtnWrapper, { Button, Icon } from './mailComposeBtn.style';

export default class ComposeBtn extends Component {
  render() {
    return (
      <MailComposeBtnWrapper>
        <Button
          onClick={event => {
            this.props.changeComposeMail(true);
            if (this.props.onDrawerClose) {
              this.props.onDrawerClose();
            }
          }}
        >
          <Icon>add</Icon>
        </Button>
      </MailComposeBtnWrapper>
    );
  }
}
