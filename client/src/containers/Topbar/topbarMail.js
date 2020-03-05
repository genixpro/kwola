import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { Link } from 'react-router-dom';
import { connect } from 'react-redux';
import IntlMessages from '../../components/utility/intlMessages';
import { TopbarDropdown } from './topbarDropdown.style';

const demoMails = [
  {
    id: 1,
    name: 'David Doe',
    time: '3 minutes ago',
    desc:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 2,
    name: 'Navis Doe',
    time: '4 minutes ago',
    desc:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 3,
    name: 'Emanual Doe',
    time: '5 minutes ago',
    desc:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 4,
    name: 'Dowain Doe',
    time: '6 minutes ago',
    desc:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
];

class TopbarMail extends Component {
  state = {
    visible: false,
    anchorEl: null,
  };
  hide = () => {
    this.setState({ visible: false });
  };
  handleVisibleChange = () => {
    this.setState({
      visible: !this.state.visible,
      anchorEl: findDOMNode(this.button),
    });
  };
  render() {
    const { url } = this.props;
    return (
      <TopbarDropdown className="topbarMail">
        <div className="dropdownBody">
          {demoMails.map(mail => (
            <Link to={`${url}/email`} onClick={this.hide} key={mail.id}>
              <div className="dropdownListItem">
                <div className="listHead">
                  <h5>{mail.name}</h5>
                  <span className="date">{mail.time}</span>
                </div>
                <p>{mail.desc}</p>
              </div>
            </Link>
          ))}
        </div>
        <a href="#!" className="viewAllBtn">
          <IntlMessages id="topbar.viewAll" />
        </a>
      </TopbarDropdown>
    );
  }
}

export default connect(state => ({
  ...state.App,
  customizedTheme: state.ThemeSwitcher.topbarTheme,
}))(TopbarMail);
