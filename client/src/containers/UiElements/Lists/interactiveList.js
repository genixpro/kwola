import React from 'react';
import PropTypes from 'prop-types';
import Lists, {
  ListItem,
  ListItemAvatar,
  ListItemIcon,
  ListItemSecondaryAction,
  ListItemText,
} from '../../../components/uielements/lists';
import Avatar from '../../../components/uielements/avatars/';
import IconButton from '../../../components/uielements/iconbutton/';
// import { FormGroup, FormControlLabel } from './lists.style';
import {
  FormGroup,
  FormControlLabel,
} from '../../../components/uielements/form/';
import Checkbox from '../../../components/uielements/checkbox/';
import Grids from '../../../components/uielements/grid/';
import Typography from '../../../components/uielements/typography/index.js';
import Icon from '../../../components/uielements/icon/index.js';

function generate(element) {
  return [0, 1, 2].map(value =>
    React.cloneElement(element, {
      key: value,
    })
  );
}

class InteractiveList extends React.Component {
  state = {
    dense: false,
    secondary: false,
  };

  render() {
    const { classes } = this.props;
    const { dense, secondary } = this.state;

    return (
      <div>
        <FormGroup row>
          <FormControlLabel
            control={
              <Checkbox
                checked={dense}
                onChange={(event, checked) => this.setState({ dense: checked })}
                value="dense"
              />
            }
            label="Enable dense"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={secondary}
                onChange={(event, checked) =>
                  this.setState({ secondary: checked })
                }
                value="secondary"
              />
            }
            label="Enable secondary text"
          />
        </FormGroup>
        <Grids container>
          <Grids item xs={12} md={6}>
            <Typography variant="h6" className={classes.title}>
              Text only
            </Typography>
            <div className={classes.demo}>
              <Lists dense={dense}>
                {generate(
                  <ListItem button>
                    <ListItemText
                      primary="Single-line item"
                      secondary={secondary ? 'Secondary text' : null}
                    />
                  </ListItem>
                )}
              </Lists>
            </div>
          </Grids>
          <Grids item xs={12} md={6}>
            <Typography variant="h6" className={classes.title}>
              Icon with text
            </Typography>
            <div className={classes.demo}>
              <Lists dense={dense}>
                {generate(
                  <ListItem button>
                    <ListItemIcon>
                      <Icon>folder</Icon>
                    </ListItemIcon>
                    <ListItemText
                      primary="Single-line item"
                      secondary={secondary ? 'Secondary text' : null}
                    />
                  </ListItem>
                )}
              </Lists>
            </div>
          </Grids>
        </Grids>
        <Grids container>
          <Grids item xs={12} md={6}>
            <Typography variant="h6" className={classes.title}>
              Avatar with text
            </Typography>
            <div className={classes.demo}>
              <Lists dense={dense}>
                {generate(
                  <ListItem button>
                    <ListItemAvatar>
                      <Avatar>
                        <Icon>folder</Icon>
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Single-line item"
                      secondary={secondary ? 'Secondary text' : null}
                    />
                  </ListItem>
                )}
              </Lists>
            </div>
          </Grids>
          <Grids item xs={12} md={6}>
            <Typography variant="h6" className={classes.title}>
              Avatar with text and icon
            </Typography>
            <div className={classes.demo}>
              <Lists dense={dense}>
                {generate(
                  <ListItem button>
                    <ListItemAvatar>
                      <Avatar>
                        <Icon>folder</Icon>
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Single-line item"
                      secondary={secondary ? 'Secondary text' : null}
                    />
                    <ListItemSecondaryAction>
                      <IconButton aria-label="Delete">
                        <Icon>delete</Icon>
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                )}
              </Lists>
            </div>
          </Grids>
        </Grids>
      </div>
    );
  }
}

InteractiveList.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default InteractiveList;
